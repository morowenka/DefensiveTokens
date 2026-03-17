"""Step 1: Generate self-labeled dataset.

Run base model on Cleaned Alpaca to produce target responses.
Uses vLLM for fast batch inference if available, else HF generate.
"""

import logging
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import setup_script
from src.dataset import load_alpaca_cleaned, save_jsonl
from src.model import format_prompt

logger = logging.getLogger(__name__)


def self_label_with_vllm(config):
    from vllm import LLM, SamplingParams

    model_name = config["model"]["name"]
    output_path = config["dataset"]["self_labeled_path"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    logger.info("Loading model with vLLM: %s", model_name)
    llm = LLM(
        model=model_name,
        dtype=config["model"]["dtype"],
        tensor_parallel_size=4,
        max_model_len=2048,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    ds = load_alpaca_cleaned()

    prompts = []
    records = []
    for sample in ds:
        instruction = sample["instruction"]
        data = sample.get("input", "")
        prompt = format_prompt(tokenizer, instruction, data if data.strip() else None)
        prompts.append(prompt)
        records.append({"instruction": instruction, "data": data})

    logger.info("Generating responses for %d samples...", len(prompts))
    outputs = llm.generate(prompts, sampling_params)

    for record, output in zip(records, outputs):
        record["base_response"] = output.outputs[0].text

    save_jsonl(records, output_path)
    logger.info("Self-labeling complete: %s", output_path)


def self_label_with_hf(config):
    from src.generation import generate_response
    from src.model import load_model_and_tokenizer

    model_name = config["model"]["name"]
    output_path = config["dataset"]["self_labeled_path"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    model, tokenizer = load_model_and_tokenizer(model_name, config["model"]["dtype"])

    ds = load_alpaca_cleaned()
    records = []

    for sample in tqdm(ds, desc="Self-labeling"):
        instruction = sample["instruction"]
        data = sample.get("input", "")
        prompt = format_prompt(tokenizer, instruction, data if data.strip() else None)
        response = generate_response(model, tokenizer, prompt, max_new_tokens)

        records.append({
            "instruction": instruction,
            "data": data,
            "base_response": response,
        })

    save_jsonl(records, output_path)
    logger.info("Self-labeling complete: %s", output_path)


def main():
    config = setup_script("self_label")

    try:
        import vllm
        logger.info("vLLM available, using fast batch inference")
        self_label_with_vllm(config)
    except ImportError:
        logger.info("vLLM not available, using HuggingFace transformers")
        self_label_with_hf(config)


if __name__ == "__main__":
    main()
