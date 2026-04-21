import os
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()


def load_model(model_id: str):
    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype="auto" if device.type == "cuda" else None,
        token=hf_token,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def classify_texts(texts, model, tokenizer, device, max_length=512, batch_size=16):
    if isinstance(texts, str):
        texts = [texts]

    results = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).detach().cpu()
            for text, row_prob in zip(batch, probs):
                top_idx = int(torch.argmax(row_prob).item())
                top_score = float(row_prob[top_idx].item())
                label = "safe" if top_idx == 0 else "unsafe"
                results.append({"text": text, "label": label, "score": top_score})
    return results
