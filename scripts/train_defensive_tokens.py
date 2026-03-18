"""Step 3: Train DefensiveTokens.

Creates a trainable prefix, freezes model, trains only prefix embeddings.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import setup_script
from src.dataset import load_jsonl
from src.model import load_model_and_tokenizer
from src.training import train_defensive_tokens

logger = logging.getLogger(__name__)


def main():
    config = setup_script("training")

    model, tokenizer = load_model_and_tokenizer(
        config["model"]["name"], config["model"]["dtype"]
    )

    samples = load_jsonl(config["dataset"]["defensive_path"])
    logger.info("Loaded %d defensive training samples", len(samples))

    train_defensive_tokens(model, tokenizer, samples, config)


if __name__ == "__main__":
    main()
