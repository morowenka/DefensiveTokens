"""DefensiveTokens training: freeze model, train only defensive token embeddings."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from src.model import (
    format_prompt,
    add_defensive_tokens,
    freeze_model_except_defensive_tokens,
    save_defensive_tokens,
)

logger = logging.getLogger(__name__)


class DefensiveDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = sample["instruction"]
        data = sample.get("data", "")
        target = sample["base_response"]

        prompt = format_prompt(
            self.tokenizer, instruction,
            data if data.strip() else None,
            use_defensive_tokens=True,
        )
        full_text = prompt + target + self.tokenizer.eos_token

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Only compute loss on response tokens
        prompt_encoded = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True
        )
        prompt_len = len(prompt_encoded["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_defensive_tokens(model, tokenizer, samples, config):
    """Train DefensiveTokens: add special tokens, freeze model, train embeddings."""
    num_tokens = config["training"]["num_defensive_tokens"]
    lr = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    max_length = config["training"]["max_length"]
    output_dir = config["training"]["output_dir"]

    token_ids = add_defensive_tokens(model, tokenizer, num_tokens)
    freeze_model_except_defensive_tokens(model, token_ids)

    dataset = DefensiveDataset(samples, tokenizer, max_length)
    use_mps = torch.backends.mps.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if use_mps else 4,
        pin_memory=not use_mps,
    )

    optimizer = torch.optim.SGD(
        [model.get_input_embeddings().weight],
        lr=lr,
    )
    total_steps = (len(dataloader) // grad_accum) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()

            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(
                    "Epoch %d, Step %d/%d, Loss: %.4f",
                    epoch + 1, step + 1, len(dataloader), avg_loss,
                )

        avg_loss = total_loss / len(dataloader)
        logger.info("Epoch %d finished. Average loss: %.4f", epoch + 1, avg_loss)

    # Handle remaining gradients
    if len(dataloader) % grad_accum != 0:
        optimizer.step()

    save_defensive_tokens(model, tokenizer, token_ids, output_dir)
    return model, token_ids
