"""Shared generation utility."""

import torch


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model given a formatted prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).lstrip()
