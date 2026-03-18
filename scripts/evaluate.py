"""Step 4: Evaluate all defenses on AlpacaFarm.

Runs: None, Reminder, Sandwich, DefensiveToken
Against: Ignore, Completion, Ignore-Completion attacks
Reports ASR table.
"""

import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import setup_script
from src.dataset import load_jsonl
from src.evaluation import evaluate_defense, print_results_table, save_results
from src.model import DefensivePrefix, load_model_and_tokenizer

logger = logging.getLogger(__name__)


def main():
    config = setup_script("evaluation")

    eval_samples = load_jsonl(config["evaluation"]["alpaca_farm_path"])
    logger.info("Loaded %d eval samples", len(eval_samples))

    model_name = config["model"]["name"]
    dtype = config["model"]["dtype"]

    # Evaluate baselines: None, Reminder, Sandwich
    model, tokenizer = load_model_and_tokenizer(model_name, dtype)
    all_results = []

    # for defense_name in ["none", "reminder", "sandwich"]:
    #     result = evaluate_defense(model, tokenizer, eval_samples, defense_name, config)
    #     all_results.append(result)

    # Free memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Evaluate DefensiveToken
    model, tokenizer = load_model_and_tokenizer(model_name, dtype)
    device = next(model.parameters()).device
    prefix = DefensivePrefix.load(config["training"]["output_dir"]).to(device)
    model.eval()

    dt_result = evaluate_defense(
        model, tokenizer, eval_samples, "defensive_token", config,
        prefix=prefix,
    )
    all_results.append(dt_result)

    print_results_table(all_results)
    save_results({"results": all_results}, config["evaluation"]["results_dir"])


if __name__ == "__main__":
    main()
