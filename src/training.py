"""DefensiveTokens training: freeze model, train only prefix embeddings."""

import logging

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from src.model import DefensivePrefix, format_prompt

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
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_defensive_tokens(model, tokenizer, samples, config):
    """Train DefensiveTokens: create prefix, freeze model, train prefix embeddings."""
    num_tokens = config["training"]["num_defensive_tokens"]
    lr = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    max_length = config["training"]["max_length"]
    output_dir = config["training"]["output_dir"]

    # Freeze entire model
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    dtype = next(model.parameters()).dtype

    prefix = DefensivePrefix(hidden_size, num_tokens, dtype=dtype).to(device)

    trainable = sum(p.numel() for p in prefix.parameters())
    logger.info("Trainable parameters: %d (prefix only)", trainable)

    dataset = DefensiveDataset(samples, tokenizer, max_length)
    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    optimizer = torch.optim.SGD([prefix.prefix], lr=lr)
    total_steps = (len(dataloader) // grad_accum) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    embed_layer = model.get_input_embeddings()
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training DefensiveTokens", ):
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            base_embeds = embed_layer(input_ids)
            inputs_embeds, attention_mask, labels = prefix.prepend(
                base_embeds, attention_mask, labels,
            )

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            if epoch == 0 and step == 0:
                updated = sum(
                    p.numel() for p in model.parameters() if p.grad is not None and p.grad.any()
                ) + sum(
                    p.numel() for p in prefix.parameters() if p.grad is not None
                )
                logger.info("Actual trainable parameters with gradients: %d", updated)

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            logger.info(
                "Epoch %d, Step %d/%d, Loss: %.4f",
                epoch + 1, step + 1, len(dataloader), outputs.loss.item(),
            )

        logger.info("Epoch %d finished", epoch + 1)

    # Handle remaining gradients
    if len(dataloader) % grad_accum != 0:
        optimizer.step()

    prefix.save(output_dir)
    return prefix
