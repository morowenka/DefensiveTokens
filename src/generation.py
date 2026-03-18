"""Shared generation utility."""

import torch


def generate_response(model, tokenizer, prompt, max_new_tokens=256, prefix=None):
    """Generate a response from the model given a formatted prompt.

    If prefix (DefensivePrefix) is provided, prepends trainable prefix
    embeddings to the input via inputs_embeds.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        if prefix is not None:
            base_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, _ = prefix.prepend(
                base_embeds, attention_mask,
            )
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_ids = outputs[0][inputs_embeds.shape[1]:]
        else:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            response_ids = outputs[0][input_ids.shape[1]:]

    return tokenizer.decode(response_ids, skip_special_tokens=True).lstrip()
