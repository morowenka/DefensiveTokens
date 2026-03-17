#!/bin/bash
# Full pipeline: self-label -> defensive dataset -> train -> evaluate
# Run from llm_def/ directory on server with GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 0: Prepare AlpacaFarm eval dataset ==="
python scripts/prepare_alpaca_farm_eval.py

echo "=== Step 1: Self-label Cleaned Alpaca with base model ==="
python scripts/self_label.py

echo "=== Step 2: Build defensive training dataset ==="
python scripts/build_defensive_dataset.py

echo "=== Step 3: Train DefensiveTokens ==="
python scripts/train_defensive_tokens.py

echo "=== Step 4: Evaluate all defenses ==="
python scripts/evaluate.py

echo "=== Done ==="
