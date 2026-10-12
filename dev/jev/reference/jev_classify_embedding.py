#!/usr/bin/env python3
"""Apply AutoTrust JEV Gemma-4's 24-slot decision head to a llama-server /embedding response.

This is an offline scoring wrapper for the *noul* (false/true) primitive.
It does not validate that the embedding equals the Transformers reference hidden state.

Usage:
  python jev_classify_embedding.py embedding.json ./jev-head
"""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("embedding", type=Path, help="Saved /embedding JSON response")
    ap.add_argument("jev_head_dir", type=Path, help="Directory with head.safetensors, judge_config.json and calibration_gold.json")
    args = ap.parse_args()

    data = json.loads(args.embedding.read_text())
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("Expected one /embedding result (a JSON list of length 1)")

    h = np.asarray(data[0]["embedding"], dtype=np.float32)
    if h.ndim == 2 and h.shape[0] == 1:
        h = h[0]
    if h.shape != (2816,):
        raise ValueError(f"Expected one 2816-dimensional vector, got {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("Embedding contains non-finite values")

    head = load_file(str(args.jev_head_dir / "head.safetensors"))
    W = np.asarray(head["proj.weight"], dtype=np.float32)
    b = np.asarray(head["proj.bias"], dtype=np.float32)
    if W.shape != (24, 2816) or b.shape != (24,):
        raise ValueError(f"Unexpected JEV head dimensions: W={W.shape}, b={b.shape}")

    cfg = json.loads((args.jev_head_dir / "judge_config.json").read_text())
    calibration = json.loads((args.jev_head_dir / "calibration_gold.json").read_text())
    start, end = cfg["slots"]["ranges"]["noul"]
    if end - start != 2:
        raise ValueError(f"Expected 2 noul slots, got {start}:{end}")
    temperature = float(calibration["per_kind"]["noul"])
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # JEV reference: softcap the 24 decision logits *before* option masking,
    # temperature scaling, and softmax.
    z = 30.0 * np.tanh((W @ h + b) / 30.0)
    selected = z[start:end] / temperature
    exponents = np.exp(selected - np.max(selected))
    probabilities = exponents / np.sum(exponents)

    print(json.dumps({
        "false": float(probabilities[0]),
        "true": float(probabilities[1]),
        "decision": "true" if probabilities[1] >= probabilities[0] else "false",
        "temperature": temperature,
        "hidden_state_l2_norm": float(np.linalg.norm(h)),
    }, indent=2))


if __name__ == "__main__":
    main()
