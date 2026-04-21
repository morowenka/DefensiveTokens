"""
HW-4: Red-teaming HiveTrace Guard Model

Usage:
  python main.py                         # full pipeline
  python main.py --phase baseline        # baseline only
  python main.py --phase attacks         # attacks only (requires model)
  python main.py --category A            # single attack category
"""

import argparse
import yaml
from tqdm import tqdm

from src.evaluator import load_model, classify_texts
from src.dataset import BASELINE_UNSAFE, BASELINE_SAFE
from src.attacks import (
    attack_unicode,
    attack_word_level,
    attack_encoding,
    attack_multilingual,
    attack_semantic,
)
from src.report import save_csv, compute_asr, generate_report


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_baseline(cfg, model, tokenizer, device):
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline evaluation")
    print(f"{'='*60}")

    unsafe_texts = [r["text"] for r in BASELINE_UNSAFE]
    print(f"Classifying {len(unsafe_texts)} unsafe baseline examples...")
    results = classify_texts(unsafe_texts, model, tokenizer, device,
                             batch_size=cfg["batch_size"])

    rows = []
    for orig, res in zip(BASELINE_UNSAFE, results):
        rows.append({
            "original": orig["text"],
            "text": orig["text"],
            "category": orig["category"],
            "attack_type": "baseline",
            "attack_subtype": "none",
            "label": res["label"],
            "score": res["score"],
            "bypass": res["label"] == "safe",
            "is_safe_baseline": False,
        })

    print(f"Classifying {len(BASELINE_SAFE)} safe baseline examples...")
    safe_results = classify_texts(BASELINE_SAFE, model, tokenizer, device,
                                  batch_size=cfg["batch_size"])
    for text, res in zip(BASELINE_SAFE, safe_results):
        rows.append({
            "original": text,
            "text": text,
            "category": "safe_baseline",
            "attack_type": "baseline",
            "attack_subtype": "none",
            "label": res["label"],
            "score": res["score"],
            "bypass": False,
            "is_safe_baseline": True,
        })

    detected = sum(1 for r in rows if not r["is_safe_baseline"] and r["label"] == "unsafe")
    total_unsafe = len(BASELINE_UNSAFE)
    print(f"Baseline: {detected}/{total_unsafe} unsafe examples correctly detected")

    fp = sum(1 for r in rows if r["is_safe_baseline"] and r["label"] == "unsafe")
    print(f"False positives: {fp}/{len(BASELINE_SAFE)}")

    return rows


def run_attacks(cfg, model, tokenizer, device, categories=None):
    print(f"\n{'='*60}")
    print("PHASE 2: Attack evaluation")
    print(f"{'='*60}")

    seeds = [r["text"] for r in BASELINE_UNSAFE]

    all_attacks = {
        "A": ("A_unicode", attack_unicode),
        "B": ("B_word_level", attack_word_level),
        "C": ("C_encoding", attack_encoding),
        "D": ("D_multilingual", attack_multilingual),
        "E": ("E_semantic", attack_semantic),
    }

    if categories:
        all_attacks = {k: v for k, v in all_attacks.items() if k in categories}

    all_variants = []
    for cat_key, (cat_name, attack_fn) in all_attacks.items():
        variants = attack_fn(seeds)
        print(f"Category {cat_key} ({cat_name}): {len(variants)} variants generated")
        all_variants.extend(variants)

    print(f"\nTotal variants to classify: {len(all_variants)}")
    variant_texts = [v[1] for v in all_variants]

    print("Running classification...")
    results = classify_texts(variant_texts, model, tokenizer, device,
                             batch_size=cfg["batch_size"])

    rows = []
    for (orig, variant, attack_type, subtype), res in zip(all_variants, results):
        rows.append({
            "original": orig,
            "text": variant,
            "category": "",
            "attack_type": attack_type,
            "attack_subtype": subtype,
            "label": res["label"],
            "score": res["score"],
            "bypass": res["label"] == "safe",
            "is_safe_baseline": False,
        })

    # Print per-category summary
    asr = compute_asr(rows)
    from collections import defaultdict
    print("\nAttack Success Rate:")
    for cat, s in sorted(asr.items(), key=lambda x: x[1]["asr"]):
        print(f"  {cat}: {s['bypass']}/{s['total']} bypasses = {s['asr']:.1%} ASR")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "attacks", "full"], default="full")
    parser.add_argument("--category", nargs="*", choices=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    cfg = load_config()
    print(f"Loading model: {cfg['model_id']}")
    model, tokenizer, device = load_model(cfg["model_id"])
    print(f"Model loaded on {device}")

    baseline_rows = []
    attack_rows = []

    if args.phase in ("baseline", "full"):
        baseline_rows = run_baseline(cfg, model, tokenizer, device)

    if args.phase in ("attacks", "full"):
        categories = args.category if args.category else ["A", "B", "C", "D", "E"]
        attack_rows = run_attacks(cfg, model, tokenizer, device, categories)

    # Save results
    all_rows = baseline_rows + attack_rows
    save_csv(all_rows, cfg["results_csv"])
    print(f"\nResults saved to {cfg['results_csv']}")

    # Generate report
    asr = compute_asr([r for r in attack_rows if not r.get("is_safe_baseline")])
    generate_report(
        baseline_rows=baseline_rows,
        attack_rows=attack_rows,
        asr=asr,
        report_path=cfg["report_md"],
    )


if __name__ == "__main__":
    main()
