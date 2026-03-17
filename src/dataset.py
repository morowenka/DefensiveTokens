"""Dataset loading and preparation utilities."""

import json
import logging
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset

from src.attacks import train_ignore_attack, train_completion_attack

logger = logging.getLogger(__name__)


def load_alpaca_cleaned():
    """Load Cleaned Alpaca dataset from HuggingFace."""
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    logger.info("Loaded %d samples from alpaca-cleaned", len(ds))
    return ds


def load_alpaca_farm_eval():
    """Load AlpacaFarm evaluation set (davinci_003_outputs equivalent).

    Returns 805 samples. For security eval, filter to those with non-empty input.
    """
    ds = load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation", split="eval")
    logger.info("Loaded %d AlpacaFarm eval samples", len(ds))
    return ds


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_defensive_dataset(self_labeled_path, output_path, injection_ratio=0.5, seed=42):
    """Build defensive training dataset from self-labeled data.

    50% clean, 50% attacked (half ignore, half completion).
    Target always remains the original model response.
    """
    np.random.seed(seed)
    random.seed(seed)
    samples = load_jsonl(self_labeled_path)
    logger.info("Building defensive dataset from %d self-labeled samples", len(samples))

    # Only inject into samples that have non-empty data field
    has_data = [s for s in samples if s.get("data", "").strip()]
    no_data = [s for s in samples if not s.get("data", "").strip()]

    num_to_inject = int(len(has_data) * injection_ratio)

    random.shuffle(has_data)
    injected_samples = has_data[:num_to_inject]
    clean_data_samples = has_data[num_to_inject:]

    defensive_samples = []

    # Clean samples (no data + clean data samples)
    for s in no_data:
        record = s.copy()
        record["is_injected"] = False
        defensive_samples.append(record)

    for s in clean_data_samples:
        record = s.copy()
        record["is_injected"] = False
        defensive_samples.append(record)

    # Injected samples: alternate between ignore and completion
    for i, s in enumerate(injected_samples):
        record = s.copy()
        original_data = s.get("data", "")
        if i % 2 == 0:
            record["data"] = train_ignore_attack(original_data, samples)
        else:
            record["data"] = train_completion_attack(
                original_data, samples, reference_outputs=s.get("base_response", "")
            )
        record["is_injected"] = True
        defensive_samples.append(record)

    random.shuffle(defensive_samples)
    save_jsonl(defensive_samples, output_path)

    injected_count = sum(1 for s in defensive_samples if s["is_injected"])
    logger.info(
        "Defensive dataset: %d total, %d injected, %d clean",
        len(defensive_samples), injected_count, len(defensive_samples) - injected_count,
    )
    return defensive_samples
