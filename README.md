# DefensiveTokens

Implementation of "Defending Against Prompt Injection With a Few DefensiveTokens" -- a test-time defense against prompt injection attacks. Special tokens with optimized embeddings are prepended to LLM input, trained to make the model ignore injected instructions in untrusted data.

## How it works

1. **Self-label** -- run base model on Alpaca dataset to generate target responses
2. **Build dataset** -- 50% clean samples + 50% with prompt injection in data field (ignore-style and completion-style attacks). Target remains the original response
3. **Train** -- add N new tokens to vocabulary, freeze all model weights, train only the defensive token embeddings via SGD
4. **Evaluate** -- measure Attack Success Rate (ASR) across attack variants with and without defenses

## Project structure

```
scripts/                  Pipeline entry points (run sequentially)
  prepare_alpaca_farm_eval.py   Step 0: filter AlpacaFarm to samples with input
  self_label.py                 Step 1: generate target responses (vLLM or HF)
  build_defensive_dataset.py    Step 2: inject attacks into training data
  train_defensive_tokens.py     Step 3: train token embeddings
  evaluate.py                   Step 4: evaluate all defenses

src/                      Core modules
  model.py                Model/tokenizer loading, defensive token management, prompt formatting
  dataset.py              Dataset loading (Alpaca, AlpacaFarm), JSONL I/O, dataset builder
  attacks.py              Prompt injection templates (ignore, completion, combined)
  defenses.py             Baseline defenses (none, reminder, sandwich)
  training.py             Training loop with gradient masking
  evaluation.py           ASR computation and results reporting
  generation.py           Shared inference utility
  common.py               Script boilerplate (logging, config, path resolution)

config.yaml               All hyperparameters and paths
run_all.sh                Run full pipeline
```

## Setup

```bash
uv pip install torch transformers datasets tqdm pyyaml
# Optional: uv pip install vllm  (for fast batch inference in step 1)
```

## Usage

Run the full pipeline:
```bash
bash run_all.sh
```

Or individual steps:
```bash
uv run python scripts/prepare_alpaca_farm_eval.py
uv run python scripts/self_label.py
uv run python scripts/build_defensive_dataset.py
uv run python scripts/train_defensive_tokens.py
uv run python scripts/evaluate.py
```

## Configuration

All parameters are in `config.yaml`:

| Section    | Key                  | Description                          |
|------------|----------------------|--------------------------------------|
| model      | name                 | HuggingFace model ID                 |
| model      | dtype                | Precision (bfloat16/float16/float32) |
| dataset    | injection_ratio      | Fraction of attacked samples (0.5)   |
| training   | num_defensive_tokens | Number of special tokens to add      |
| training   | learning_rate        | SGD learning rate for embeddings     |
| training   | num_epochs           | Training epochs                      |
| evaluation | max_new_tokens       | Generation length for evaluation     |

## Evaluated defenses

| Defense         | Method                                              |
|-----------------|-----------------------------------------------------|
| none            | No defense (baseline)                               |
| reminder        | Append "do not follow" warning to instruction       |
| sandwich        | Append task reminder after untrusted data            |
| defensive_token | Prepend trained DefensiveTokens to input             |

## Attacks

- **ignore** -- social engineering to disregard prior instructions
- **completion** -- fake response/instruction delimiters in data
- **ignore_completion** -- combined ignore + completion

Metric: max ASR across attack variants (lower is better).
