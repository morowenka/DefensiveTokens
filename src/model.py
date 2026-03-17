import json
import logging
from pathlib import Path

import torch
import transformers

logger = logging.getLogger(__name__)


def _build_defensive_token_names(num_tokens):
    return [f"[DefensiveToken{i}]" for i in range(num_tokens)]


def _build_chat_template(num_tokens):
    token_str = "".join(f"[DefensiveToken{i}]" for i in range(num_tokens))
    return (
        "{%- if add_defensive_tokens %}\n"
        "{{- '" + token_str + "' }}\n"
        "{%- endif %}\n"
        "{%- for message in messages %}\n"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '\\n\\n<|im_end|>\\n' }}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n"
        "{{- '<|im_start|>assistant\\n' }}\n"
        "{%- endif %}\n"
    )

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


def add_defensive_tokens(model, tokenizer, num_tokens=5):
    """Add DefensiveToken special tokens to tokenizer and resize model embeddings.

    Returns the indices of the new tokens.
    """
    token_names = _build_defensive_token_names(num_tokens)
    num_new = tokenizer.add_special_tokens(
        {"additional_special_tokens": token_names}
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize defensive token embeddings with Gaussian N(0, I) as per paper
    embed_weight = model.get_input_embeddings().weight.data
    embed_dim = embed_weight.shape[1]
    token_ids = [tokenizer.convert_tokens_to_ids(name) for name in token_names]
    for tid in token_ids:
        embed_weight[tid] = torch.randn(embed_dim, dtype=embed_weight.dtype)

    logger.info("Added %d defensive tokens with Gaussian init", num_new)

    tokenizer.chat_template = _build_chat_template(num_tokens)

    return token_ids


def freeze_model_except_defensive_tokens(model, token_ids):
    """Freeze all parameters except the defensive token embeddings."""
    for param in model.parameters():
        param.requires_grad = False

    embed_weight = model.get_input_embeddings().weight
    embed_weight.requires_grad = True

    # We will use a hook to zero out gradients for all tokens except defensive ones
    def zero_non_defensive_grads(grad):
        mask = torch.zeros_like(grad)
        for tid in token_ids:
            mask[tid] = 1.0
        return grad * mask

    embed_weight.register_hook(zero_non_defensive_grads)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d (only defensive token embeddings are updated)", trainable)


def save_defensive_tokens(model, tokenizer, token_ids, output_dir):
    """Save only the defensive token embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_names = _build_defensive_token_names(len(token_ids))
    embed_weight = model.get_input_embeddings().weight.data
    tokens_data = {}
    for name, tid in zip(token_names, token_ids):
        tokens_data[name] = embed_weight[tid].cpu().float().tolist()

    with open(output_dir / "defensive_tokens.json", "w") as f:
        json.dump(tokens_data, f)

    tokenizer.save_pretrained(output_dir)
    logger.info("Saved defensive tokens to %s", output_dir)


def load_defensive_tokens(model, tokenizer, checkpoint_dir):
    """Load trained defensive token embeddings into model."""
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "defensive_tokens.json") as f:
        tokens_data = json.load(f)

    num_tokens = len(tokens_data)
    token_ids = add_defensive_tokens(model, tokenizer, num_tokens)
    token_names = _build_defensive_token_names(num_tokens)

    embed_weight = model.get_input_embeddings().weight.data
    for name, tid in zip(token_names, token_ids):
        embed_weight[tid] = torch.tensor(tokens_data[name], dtype=embed_weight.dtype)

    logger.info("Loaded defensive tokens from %s", checkpoint_dir)
    return token_ids


def recursive_filter(text, filters=None):
    """Remove all special/filtered tokens from untrusted text."""
    if filters is None:
        filters = FILTERED_BASE_TOKENS + _build_defensive_token_names(10)
    orig = text
    for f in filters:
        text = text.replace(f, "")
    if text != orig:
        return recursive_filter(text, filters)
    return text


def format_prompt(tokenizer, instruction, data=None, use_defensive_tokens=False):
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
        add_defensive_tokens=use_defensive_tokens,
    )
    return prompt
