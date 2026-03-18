import logging
from pathlib import Path

import torch
import torch.nn as nn
import transformers

logger = logging.getLogger(__name__)

FILTERED_BASE_TOKENS = ["[INST]", "[INPT]", "[RESP]", "[MARK]", "[COLN]", "##"]


def load_model_and_tokenizer(model_name, dtype="bfloat16"):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    logger.info("Loading model: %s", model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


class DefensivePrefix(nn.Module):
    """Trainable prefix embeddings prepended to model input.

    This is the core of the DefensiveToken approach: a small set of
    learnable vectors that are concatenated before the input embeddings.
    The rest of the model stays frozen.
    """

    def __init__(self, hidden_size, num_tokens=5, dtype=torch.bfloat16):
        super().__init__()
        self.num_tokens = num_tokens
        self.prefix = nn.Parameter(
            torch.randn(num_tokens, hidden_size, dtype=dtype)
        )

    def forward(self, batch_size):
        return self.prefix.unsqueeze(0).expand(batch_size, -1, -1)

    def prepend(self, base_embeds, attention_mask, labels=None):
        """Prepend prefix embeddings and extend attention mask / labels."""
        batch_size = base_embeds.size(0)
        device = base_embeds.device
        prefix_embeds = self.forward(batch_size)
        inputs_embeds = torch.cat([prefix_embeds, base_embeds], dim=1)

        prefix_mask = torch.ones(
            batch_size, self.num_tokens,
            dtype=attention_mask.dtype, device=device,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, self.num_tokens),
                -100, dtype=labels.dtype, device=device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.prefix.data, output_dir / "defensive_prefix.pt")
        logger.info("Saved defensive prefix to %s", output_dir)

    @classmethod
    def load(cls, checkpoint_dir, device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        prefix_data = torch.load(
            checkpoint_dir / "defensive_prefix.pt",
            map_location=device,
            weights_only=True,
        )
        num_tokens, hidden_size = prefix_data.shape
        instance = cls(hidden_size, num_tokens, dtype=prefix_data.dtype)
        instance.prefix.data.copy_(prefix_data)
        logger.info("Loaded defensive prefix from %s (%d tokens)", checkpoint_dir, num_tokens)
        return instance


def recursive_filter(text, filters=None):
    """Remove all special/filtered tokens from untrusted text."""
    if filters is None:
        filters = FILTERED_BASE_TOKENS
    orig = text
    for f in filters:
        text = text.replace(f, "")
    if text != orig:
        return recursive_filter(text, filters)
    return text


def format_prompt(tokenizer, instruction, data=None):
    """Format input using the model's chat template.

    System role = trusted instruction.
    User role = untrusted data (filtered).
    """
    messages = [{"role": "system", "content": instruction}]
    if data:
        filtered_data = recursive_filter(data)
        messages.append({"role": "user", "content": filtered_data})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt
