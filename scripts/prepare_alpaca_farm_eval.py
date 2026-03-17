"""Prepare AlpacaFarm evaluation dataset.

Downloads AlpacaFarm evaluation set and filters to samples with
non-empty input field for security evaluation (~208 samples).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import setup_script
from src.dataset import load_alpaca_farm_eval, save_jsonl

logger = logging.getLogger(__name__)


def main():
    config = setup_script("prepare_eval")

    ds = load_alpaca_farm_eval()

    eval_samples = []
    for sample in ds:
        data = sample.get("input", "")
        if data.strip():
            eval_samples.append({
                "instruction": sample["instruction"],
                "input": data,
                "output": sample.get("output", ""),
            })

    logger.info("Samples with non-empty input: %d", len(eval_samples))
    save_jsonl(eval_samples, config["evaluation"]["alpaca_farm_path"])


if __name__ == "__main__":
    main()
