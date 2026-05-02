#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test action-distribution health for the self-forcing warmup pipeline.

Combines and adapts the SF-debug repo's two diagnostic scripts:
  - scripts/test_training_action_distribution.py (training-pipeline action stats)
  - scripts/test_delta_action_distribution.py    (teacher-cache action stats + comparison)

Adapted for the C-H-S-S surgical pipeline, which uses ``MixedLeRobotDataset``
with multi-dataset specs (``JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS`` for the
current JHU dVRK fine-tune; ``SRTH_PORCINE_CHOLE_FIX_TRAIN_DATASET_SPECS`` for
the upcoming chole-c-h-s-s fine-tune; etc.).

Designed to run on the cluster inside the same container that produced the
cache (cosmos-predict-2.5.sqsh -- this script does not need NATTEN). Output
is intentionally verbose and copy-paste-friendly so it can be fed back to a
chat assistant for analysis.

Three checks performed:

  1. **Training-pipeline distribution** (always): instantiates
     ``MixedLeRobotDataset`` with the requested specs and samples N actions.
     Reports global + per-dim + per-subset statistics.

  2. **Teacher cache distribution** (if ``--teacher_cache_path`` is provided):
     loads N action JSON files written by ``inference_jhu_dvrk_warmup.py`` and
     reports the same statistics. Auto-detects which sub-dataset each cached
     index falls into.

  3. **Identity check** (if both are provided): for the first M shared indices
     (default 100), verifies cache[idx] ~= dataset[idx] within float tolerance.
     This catches data-pipeline regressions that distribution-only checks miss
     (e.g. if the transform stack changed between cache generation and now).

A "healthy" report has:
  - Global mean ~= 0 and global std ~= 1 over the active dimensions
  - Padded dimensions all == 0 (e.g. dims 20-43 for JHU dVRK Mono with
    20D real action zero-padded to MAX_ACTION_DIM=44)
  - Cache and dataset distributions match within tolerance
  - Identity check passes for all M sampled indices
  - Per-subset statistics show no extreme outliers

Usage examples:

  # JHU dVRK Mono - current run; cache lives in the repo root on the cluster
  DEBUG_NORMALIZATION=1 python scripts/test_action_distribution.py \\
      --dataset_specs JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS \\
      --teacher_cache_path datasets/jhu_dvrk_mono_warmup_4step/actions \\
      --num_samples 1000 \\
      --identity_check_count 100 \\
      --verbose

  # Same, but uniform sampling across the full mixture for representative stats
  DEBUG_NORMALIZATION=1 python scripts/test_action_distribution.py \\
      --dataset_specs JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS \\
      --num_samples 1000 \\
      --uniform

  # Future chole fine-tune (single-dataset mixture)
  DEBUG_NORMALIZATION=1 python scripts/test_action_distribution.py \\
      --dataset_specs SRTH_PORCINE_CHOLE_FIX_TRAIN_DATASET_SPECS \\
      --teacher_cache_path datasets/chole_warmup_4step/actions \\
      --num_samples 1000 \\
      --verbose
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm


# Sentinel thresholds used by the diagnostic block below. Tuned for normalized
# (mean_std) actions where a single dim's std should be ~1.0.
ACTIVE_STD_THRESHOLD = 1e-6  # below => treated as a zero-padded dim
NORMALIZED_GLOBAL_MEAN_TOL = 0.5
NORMALIZED_GLOBAL_STD_LO = 0.5
NORMALIZED_GLOBAL_STD_HI = 2.0
PER_DIM_MEAN_TOL = 1.0  # |mean| > tol => flag this dim
PER_DIM_STD_LO = 0.1
PER_DIM_STD_HI = 5.0
OUTLIER_VALUE_THRESHOLD = 10.0  # |x| > tol => count as extreme outlier
OUTLIER_PERCENT_THRESHOLD = 0.5  # > pct => flag dim
COMPARISON_MEAN_TOL = 0.05  # cache vs dataset per-dim mean diff
COMPARISON_STD_TOL = 0.05  # cache vs dataset per-dim std diff
IDENTITY_RTOL = 1e-4
IDENTITY_ATOL = 1e-5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify action distribution for the C-H-S-S self-forcing pipeline"
    )
    parser.add_argument(
        "--dataset_specs",
        type=str,
        default="JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS",
        help=(
            "Name of a dataset specs constant in groot_configs.py "
            "(e.g. JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS, "
            "SRTH_PORCINE_CHOLE_FIX_TRAIN_DATASET_SPECS, OPEN_H_DATASET_SPECS). "
            "Default: JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS."
        ),
    )
    parser.add_argument(
        "--teacher_cache_path",
        type=str,
        default=None,
        help=(
            "Path to teacher trajectory cache directory containing "
            "<idx>.json action files (e.g. datasets/jhu_dvrk_mono_warmup_4step/actions). "
            "If omitted, only the training-pipeline analysis runs."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to draw from each source (default: 1000).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=13,
        help="Frames per sample passed to MixedLeRobotDataset (default: 13).",
    )
    parser.add_argument(
        "--max_action_dim",
        type=int,
        default=None,
        help=(
            "Override MAX_ACTION_DIM. Default: import from groot_configs.MAX_ACTION_DIM "
            "(currently 44 for the JHU/CMR mixture)."
        ),
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="train",
        choices=["train", "test", "full"],
        help="Data split passed to MixedLeRobotDataset (default: train).",
    )
    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.02,
        help=(
            "Default test split ratio for sub-datasets that don't override it. "
            "Match what the inference script used (default: 0.02)."
        ),
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Random sampling of dataset indices (default: sequential).",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Uniformly spaced indices across the dataset (default: sequential).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --random sampling (default: 42).",
    )
    parser.add_argument(
        "--identity_check_count",
        type=int,
        default=100,
        help=(
            "Number of cache indices to verify against dataset[idx] for exact match "
            "(only when --teacher_cache_path is set). Default: 100. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--skip_dataset",
        action="store_true",
        help=(
            "Skip building MixedLeRobotDataset (only analyze the cache). Useful for "
            "quick cache-only checks when the lustre dataset paths are unavailable."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-dimension stats tables (else just summary stats).",
    )
    parser.add_argument(
        "--save_report",
        type=str,
        default=None,
        help="Path to save full results as JSON (for downstream analysis).",
    )
    return parser.parse_args()


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _subsection(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def load_dataset_specs(name: str) -> tuple[list[dict], int]:
    """Resolve a string like 'JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS' to a
    (specs_list, MAX_ACTION_DIM) pair pulled from ``groot_configs``."""
    try:
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams import (
            groot_configs,
        )
    except ImportError as e:
        print(f"FATAL: failed to import groot_configs: {e}")
        print("  Make sure cosmos_predict2 is on PYTHONPATH (export PYTHONPATH=/workspace:...)")
        sys.exit(1)

    if not hasattr(groot_configs, name):
        avail = [
            attr
            for attr in dir(groot_configs)
            if attr.endswith("_SPECS") and not attr.startswith("_")
        ]
        print(f"FATAL: '{name}' not found in groot_configs.")
        print(f"  Available *_SPECS constants: {avail}")
        sys.exit(1)

    specs = getattr(groot_configs, name)
    if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
        print(f"FATAL: '{name}' is not a list[dict]: type = {type(specs)}")
        sys.exit(1)

    max_action_dim = getattr(groot_configs, "MAX_ACTION_DIM", 44)

    return specs, max_action_dim


def build_mixed_dataset(
    specs: list[dict],
    num_frames: int,
    data_split: str,
    max_action_dim: int,
    test_split_ratio: float,
):
    """Instantiate ``MixedLeRobotDataset`` with the given specs."""
    from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
        MixedLeRobotDataset,
    )

    print(
        f"Building MixedLeRobotDataset: {len(specs)} subset(s), "
        f"num_frames={num_frames}, data_split={data_split}, "
        f"max_action_dim={max_action_dim}, test_split_ratio={test_split_ratio}"
    )
    return MixedLeRobotDataset(
        dataset_specs=specs,
        num_frames=num_frames,
        data_split=data_split,
        max_action_dim=max_action_dim,
        downscaled_res=False,
        test_split_ratio=test_split_ratio,
    )


def get_subset_info(dataset) -> list[dict]:
    """Extract per-subset metadata for index -> subset mapping."""
    info: list[dict] = []
    cum = list(getattr(dataset, "_cumulative_sizes", []))
    sub_datasets = getattr(dataset, "sub_datasets", [])
    embodiment_tags = getattr(dataset, "embodiment_tags", [])
    virtual_sizes = getattr(dataset, "virtual_sizes", [])
    repeat_factors = getattr(dataset, "repeat_factors", [])
    if not sub_datasets:
        return info
    for i, ds in enumerate(sub_datasets):
        path_short = Path(ds.dataset_path).name if hasattr(ds, "dataset_path") else f"sub_{i}"
        start = int(cum[i - 1]) if i > 0 else 0
        end = int(cum[i])
        info.append(
            {
                "index": i,
                "name": path_short,
                "embodiment": embodiment_tags[i] if i < len(embodiment_tags) else "?",
                "real_len": int(len(ds)),
                "virtual_size": int(virtual_sizes[i]) if i < len(virtual_sizes) else 0,
                "repeat_factor": int(repeat_factors[i]) if i < len(repeat_factors) else 0,
                "virtual_idx_start": start,
                "virtual_idx_end_excl": end,
            }
        )
    return info


def index_to_subset(idx: int, subset_info: list[dict]) -> Optional[dict]:
    for s in subset_info:
        if s["virtual_idx_start"] <= idx < s["virtual_idx_end_excl"]:
            return s
    return None


def make_indices(num_samples: int, dataset_len: int, mode: str, seed: int) -> np.ndarray:
    """Generate ``num_samples`` indices in [0, dataset_len) per the requested mode."""
    n = min(num_samples, dataset_len)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(dataset_len, size=n, replace=False)
    if mode == "uniform":
        return np.linspace(0, dataset_len - 1, n, dtype=int)
    return np.arange(n, dtype=int)  # sequential


def collect_dataset_actions(
    dataset, indices: np.ndarray, subset_info: list[dict]
) -> tuple[np.ndarray, list[Optional[int]]]:
    """Iterate the dataset at the given indices and return (actions, subset_ids).

    Each row of ``actions`` is the full per-sample action chunk (T, D) flattened
    to (T*D) on disk would be too large; we keep (chunk_T, D) and concatenate
    across samples. Returns shape (sum(chunk_T), D)."""
    sample_chunks: list[np.ndarray] = []
    sample_subsets: list[Optional[int]] = []
    failed = 0
    for idx in tqdm(indices, desc="Loading from MixedLeRobotDataset"):
        idx_int = int(idx)
        try:
            data = dataset[idx_int]
            action = data["action"]
            if hasattr(action, "numpy"):
                action = action.cpu().numpy() if hasattr(action, "cpu") else action.numpy()
            sample_chunks.append(np.asarray(action, dtype=np.float32))
            subset = index_to_subset(idx_int, subset_info)
            sample_subsets.append(subset["index"] if subset else None)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: dataset[{idx_int}] failed: {e}")
    if failed > 5:
        print(f"  WARNING: {failed - 5} more failures suppressed.")
    if not sample_chunks:
        raise RuntimeError("No samples loaded from dataset")
    all_actions = np.concatenate(sample_chunks, axis=0)
    return all_actions, sample_subsets


def collect_cache_actions(
    cache_dir: str, num_samples: int, subset_info: list[dict]
) -> tuple[np.ndarray, list[int], list[Optional[int]]]:
    """Load the first ``num_samples`` action JSON files from the cache.

    Files are expected to be named ``<idx>.json`` with content shape (T, D).
    Returns (actions_TxD_concatenated, sorted_indices, sample_subsets)."""
    cache_path = Path(cache_dir)
    json_files = sorted(
        glob.glob(str(cache_path / "*.json")),
        key=lambda p: int(Path(p).stem),
    )
    if not json_files:
        raise FileNotFoundError(f"No <idx>.json files found in {cache_dir}")
    json_files = json_files[:num_samples]

    sample_chunks: list[np.ndarray] = []
    indices: list[int] = []
    sample_subsets: list[Optional[int]] = []
    for f in tqdm(json_files, desc="Loading from teacher cache"):
        try:
            with open(f) as fh:
                arr = json.load(fh)
            arr_np = np.asarray(arr, dtype=np.float32)
            sample_chunks.append(arr_np)
            idx_int = int(Path(f).stem)
            indices.append(idx_int)
            subset = index_to_subset(idx_int, subset_info) if subset_info else None
            sample_subsets.append(subset["index"] if subset else None)
        except Exception as e:
            print(f"  WARNING: cache file {f} failed: {e}")
    if not sample_chunks:
        raise RuntimeError(f"Could not load any cache file from {cache_dir}")
    return np.concatenate(sample_chunks, axis=0), indices, sample_subsets


def detect_active_padded_dims(
    actions: np.ndarray, std_threshold: float = ACTIVE_STD_THRESHOLD
) -> tuple[list[int], list[int]]:
    """Auto-detect which dims look 'active' (std > threshold) vs zero-padded."""
    per_dim_std = actions.std(axis=0)
    active = [int(d) for d in range(actions.shape[1]) if per_dim_std[d] > std_threshold]
    padded = [int(d) for d in range(actions.shape[1]) if per_dim_std[d] <= std_threshold]
    return active, padded


def compute_full_stats(actions: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "n": int(actions.shape[0]),
        "dim": int(actions.shape[1]) if actions.ndim > 1 else 1,
        "global_mean": float(actions.mean()),
        "global_std": float(actions.std()),
        "global_min": float(actions.min()),
        "global_max": float(actions.max()),
        "per_dim_mean": actions.mean(axis=0).tolist(),
        "per_dim_std": actions.std(axis=0).tolist(),
        "per_dim_min": actions.min(axis=0).tolist(),
        "per_dim_max": actions.max(axis=0).tolist(),
        "per_dim_q01": np.percentile(actions, 1, axis=0).tolist(),
        "per_dim_q99": np.percentile(actions, 99, axis=0).tolist(),
        "global_q01": float(np.percentile(actions, 1)),
        "global_q05": float(np.percentile(actions, 5)),
        "global_q50": float(np.percentile(actions, 50)),
        "global_q95": float(np.percentile(actions, 95)),
        "global_q99": float(np.percentile(actions, 99)),
    }


def print_global_block(name: str, stats: dict[str, Any]) -> None:
    _subsection(f"GLOBAL STATS - {name}")
    print(f"  shape:        {stats['shape']}  (n_action_steps={stats['n']}, dim={stats['dim']})")
    print(f"  mean:  {stats['global_mean']:>12.6f}")
    print(f"  std:   {stats['global_std']:>12.6f}")
    print(f"  min:   {stats['global_min']:>12.6f}")
    print(f"  max:   {stats['global_max']:>12.6f}")
    print(
        "  percentiles:  "
        f"q01={stats['global_q01']:.4f}  q05={stats['global_q05']:.4f}  "
        f"q50={stats['global_q50']:.4f}  q95={stats['global_q95']:.4f}  q99={stats['global_q99']:.4f}"
    )

    is_norm = (
        abs(stats["global_mean"]) < NORMALIZED_GLOBAL_MEAN_TOL
        and NORMALIZED_GLOBAL_STD_LO < stats["global_std"] < NORMALIZED_GLOBAL_STD_HI
    )
    if is_norm:
        print("  [OK] looks normalized (mean~0, std in [0.5, 2.0])")
    else:
        print(
            f"  [WARN] does NOT look normalized "
            f"(want |mean|<{NORMALIZED_GLOBAL_MEAN_TOL}, std in "
            f"[{NORMALIZED_GLOBAL_STD_LO}, {NORMALIZED_GLOBAL_STD_HI}])"
        )


def print_active_padded_breakdown(
    name: str, actions: np.ndarray, active: list[int], padded: list[int]
) -> dict[str, Any]:
    _subsection(f"ACTIVE vs PADDED DIMS - {name}")
    print(f"  active dims (std > {ACTIVE_STD_THRESHOLD:g}): {len(active)}")
    print(f"    indices: {active}")
    print(f"  padded dims (std == 0):                       {len(padded)}")
    print(f"    indices: {padded}")

    if active:
        active_actions = actions[:, active]
        a_mean = float(active_actions.mean())
        a_std = float(active_actions.std())
        print(f"  active-only global stats:  mean={a_mean:>10.6f}  std={a_std:>10.6f}")
        if abs(a_mean) < NORMALIZED_GLOBAL_MEAN_TOL and NORMALIZED_GLOBAL_STD_LO < a_std < NORMALIZED_GLOBAL_STD_HI:
            print("  [OK] active dims look normalized (mean~0, std in [0.5, 2.0])")
        else:
            print("  [WARN] active dims do NOT look normalized")
    else:
        a_mean, a_std = float("nan"), float("nan")
        print("  [WARN] no active dims detected (all data appears zero?)")

    if padded:
        padded_actions = actions[:, padded]
        p_max_abs = float(np.max(np.abs(padded_actions)))
        print(f"  padded dims max |value|:   {p_max_abs:.2e}")
        if p_max_abs > 1e-6:
            print("  [WARN] padded dims contain non-zero values")
        else:
            print("  [OK] all padded dims are exactly zero")
    else:
        p_max_abs = 0.0

    return {
        "active_dims": active,
        "padded_dims": padded,
        "active_mean": float(a_mean),
        "active_std": float(a_std),
        "padded_max_abs": float(p_max_abs),
    }


def print_per_dim_table(name: str, actions: np.ndarray, active: list[int]) -> list[dict[str, Any]]:
    _subsection(f"PER-DIM STATS - {name}  (only first {min(actions.shape[1], 50)} dims)")
    print(
        f"  {'Dim':>4}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}  "
        f"{'Q01':>10}  {'Q99':>10}  Status"
    )
    out = []
    n_dims_shown = min(actions.shape[1], 50)
    for d in range(n_dims_shown):
        col = actions[:, d]
        mean = float(col.mean())
        std = float(col.std())
        if d in active:
            mean_ok = abs(mean) < PER_DIM_MEAN_TOL
            std_ok = PER_DIM_STD_LO < std < PER_DIM_STD_HI
            outliers = float(np.mean(np.abs(col) > OUTLIER_VALUE_THRESHOLD) * 100.0)
            issues = []
            if not mean_ok:
                issues.append(f"|mean|>{PER_DIM_MEAN_TOL}")
            if not std_ok:
                issues.append(f"std<{PER_DIM_STD_LO} or >{PER_DIM_STD_HI}")
            if outliers > OUTLIER_PERCENT_THRESHOLD:
                issues.append(f"{outliers:.1f}% outliers")
            status = "OK" if not issues else "WARN: " + "; ".join(issues)
        else:
            outliers = 0.0
            status = "padded"
        per_dim = {
            "dim": d,
            "mean": mean,
            "std": std,
            "min": float(col.min()),
            "max": float(col.max()),
            "q01": float(np.percentile(col, 1)),
            "q99": float(np.percentile(col, 99)),
            "outlier_pct": outliers,
            "status": status,
        }
        out.append(per_dim)
        print(
            f"  {d:>4}  {mean:>10.4f}  {std:>10.4f}  {per_dim['min']:>10.4f}  "
            f"{per_dim['max']:>10.4f}  {per_dim['q01']:>10.4f}  {per_dim['q99']:>10.4f}  {status}"
        )
    if actions.shape[1] > n_dims_shown:
        print(f"  ... {actions.shape[1] - n_dims_shown} more dims hidden ...")
    return out


def print_per_subset(
    name: str,
    actions: np.ndarray,
    sample_subsets: list[Optional[int]],
    chunk_size: int,
    subset_info: list[dict],
) -> dict[int, dict[str, Any]]:
    """Per-subset stats: which subset each chunk_T-row block of `actions` belongs to."""
    _subsection(f"PER-SUBSET BREAKDOWN - {name}")
    if not subset_info or not sample_subsets:
        print("  (no subset info available)")
        return {}

    by_subset: dict[Optional[int], list[int]] = {}
    for i, sid in enumerate(sample_subsets):
        by_subset.setdefault(sid, []).append(i)
    print(
        f"  {'#':>2}  {'subset':<32}  {'embodiment':<18}  {'samples':>8}  "
        f"{'%':>6}  {'mean':>10}  {'std':>10}"
    )
    out: dict[int, dict[str, Any]] = {}
    total = len(sample_subsets)
    for sid in sorted(
        by_subset.keys(), key=lambda x: (x is None, x if x is not None else -1)
    ):
        sample_idxs = by_subset[sid]
        rows = []
        for si in sample_idxs:
            rows.append(actions[si * chunk_size : (si + 1) * chunk_size])
        if not rows:
            continue
        block = np.concatenate(rows, axis=0)
        if sid is None:
            sub_name, embodiment = "<unknown>", "?"
        elif 0 <= sid < len(subset_info):
            sub_name = subset_info[sid]["name"]
            embodiment = str(subset_info[sid]["embodiment"])
        else:
            sub_name, embodiment = f"sid={sid}", "?"
        m, s = float(block.mean()), float(block.std())
        n_samples = len(sample_idxs)
        pct = 100.0 * n_samples / total
        print(
            f"  {sid if sid is not None else '?':>2}  {sub_name:<32}  "
            f"{embodiment:<18}  {n_samples:>8}  {pct:>5.1f}%  {m:>10.6f}  {s:>10.6f}"
        )
        out[sid if sid is not None else -1] = {
            "subset": sub_name,
            "embodiment": embodiment,
            "samples": n_samples,
            "pct": pct,
            "mean": m,
            "std": s,
        }
    return out


def compare_distributions(
    name1: str,
    actions1: np.ndarray,
    name2: str,
    actions2: np.ndarray,
) -> dict[str, Any]:
    _subsection(f"COMPARISON - {name1} vs {name2}")
    if actions1.shape[1] != actions2.shape[1]:
        print(
            f"  [WARN] dimension mismatch: {actions1.shape} vs {actions2.shape} - skipping per-dim comparison"
        )
        return {"skipped": True, "reason": "dim_mismatch"}

    mean1 = actions1.mean(axis=0)
    mean2 = actions2.mean(axis=0)
    std1 = actions1.std(axis=0)
    std2 = actions2.std(axis=0)
    mean_diff = np.abs(mean1 - mean2)
    std_diff = np.abs(std1 - std2)

    max_mean_dim = int(np.argmax(mean_diff))
    max_std_dim = int(np.argmax(std_diff))
    max_mean_diff = float(mean_diff[max_mean_dim])
    max_std_diff = float(std_diff[max_std_dim])
    print(
        f"  per-dim mean abs-diff:   max={max_mean_diff:.6f} (dim {max_mean_dim}),  "
        f"avg={float(mean_diff.mean()):.6f}"
    )
    print(
        f"  per-dim std abs-diff:    max={max_std_diff:.6f} (dim {max_std_dim}),  "
        f"avg={float(std_diff.mean()):.6f}"
    )

    mean_ok = max_mean_diff < COMPARISON_MEAN_TOL
    std_ok = max_std_diff < COMPARISON_STD_TOL
    if mean_ok and std_ok:
        print(
            f"  [OK] distributions match within tolerance "
            f"(mean_tol={COMPARISON_MEAN_TOL}, std_tol={COMPARISON_STD_TOL})"
        )
    else:
        if not mean_ok:
            print(
                f"  [WARN] max mean diff {max_mean_diff:.4f} exceeds tol {COMPARISON_MEAN_TOL} (dim {max_mean_dim})"
            )
        if not std_ok:
            print(
                f"  [WARN] max std diff {max_std_diff:.4f} exceeds tol {COMPARISON_STD_TOL} (dim {max_std_dim})"
            )
    return {
        "skipped": False,
        "max_mean_diff": max_mean_diff,
        "max_mean_dim": max_mean_dim,
        "max_std_diff": max_std_diff,
        "max_std_dim": max_std_dim,
        "mean_diff_per_dim": mean_diff.tolist(),
        "std_diff_per_dim": std_diff.tolist(),
        "passes_tolerance": bool(mean_ok and std_ok),
    }


def identity_check(
    dataset,
    cache_dir: str,
    max_check: int,
) -> dict[str, Any]:
    _subsection(f"IDENTITY CHECK - cache[idx] vs dataset[idx]  (first {max_check})")
    if max_check <= 0:
        print("  (skipped)")
        return {"checked": 0, "passed": 0, "skipped": True}
    cache_path = Path(cache_dir)
    json_files = sorted(
        glob.glob(str(cache_path / "*.json")),
        key=lambda p: int(Path(p).stem),
    )[:max_check]
    if not json_files:
        print(f"  [WARN] no cache files in {cache_dir}")
        return {"checked": 0, "passed": 0, "skipped": True}

    n_passed = 0
    n_failed = 0
    fail_examples: list[dict] = []
    for f in tqdm(json_files, desc="Identity check"):
        idx = int(Path(f).stem)
        try:
            with open(f) as fh:
                cached = np.asarray(json.load(fh), dtype=np.float32)
            data = dataset[idx]
            action = data["action"]
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            elif hasattr(action, "numpy"):
                action = action.numpy()
            live = np.asarray(action, dtype=np.float32)
            if cached.shape != live.shape:
                n_failed += 1
                if len(fail_examples) < 3:
                    fail_examples.append(
                        {"idx": idx, "reason": "shape", "cached_shape": list(cached.shape), "live_shape": list(live.shape)}
                    )
                continue
            ok = np.allclose(cached, live, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)
            if ok:
                n_passed += 1
            else:
                n_failed += 1
                if len(fail_examples) < 3:
                    abs_diff = float(np.max(np.abs(cached - live)))
                    fail_examples.append(
                        {"idx": idx, "reason": "value", "max_abs_diff": abs_diff}
                    )
        except Exception as e:
            n_failed += 1
            if len(fail_examples) < 3:
                fail_examples.append({"idx": idx, "reason": "exception", "error": str(e)})

    total = n_passed + n_failed
    print(
        f"  checked: {total},  passed: {n_passed},  failed: {n_failed}  "
        f"(rtol={IDENTITY_RTOL}, atol={IDENTITY_ATOL})"
    )
    if n_failed == 0 and n_passed > 0:
        print("  [OK] all sampled cache rows match the live dataset")
    elif n_failed > 0:
        print(f"  [WARN] {n_failed} mismatches detected, first few:")
        for ex in fail_examples:
            print(f"    {ex}")
    return {
        "checked": total,
        "passed": n_passed,
        "failed": n_failed,
        "failures_sample": fail_examples,
        "skipped": False,
    }


def healthy_summary(
    cache_full_stats: Optional[dict[str, Any]],
    dataset_full_stats: Optional[dict[str, Any]],
    cache_active_padded: Optional[dict[str, Any]],
    dataset_active_padded: Optional[dict[str, Any]],
    comparison: Optional[dict[str, Any]],
    identity: Optional[dict[str, Any]],
) -> None:
    """Print a single-screen verdict that's easy to copy-paste back."""
    _section("HEALTH SUMMARY  (copy-paste back to assistant for analysis)")
    checks: list[tuple[str, bool, str]] = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    if dataset_full_stats is not None:
        ds_norm = (
            abs(dataset_full_stats["global_mean"]) < NORMALIZED_GLOBAL_MEAN_TOL
            and NORMALIZED_GLOBAL_STD_LO < dataset_full_stats["global_std"] < NORMALIZED_GLOBAL_STD_HI
        )
        _check(
            "dataset global is normalized",
            ds_norm,
            f"mean={dataset_full_stats['global_mean']:.4f}, std={dataset_full_stats['global_std']:.4f}",
        )
    if dataset_active_padded is not None:
        active_norm = (
            abs(dataset_active_padded["active_mean"]) < NORMALIZED_GLOBAL_MEAN_TOL
            and NORMALIZED_GLOBAL_STD_LO < dataset_active_padded["active_std"] < NORMALIZED_GLOBAL_STD_HI
        )
        _check(
            "dataset active dims are normalized",
            active_norm,
            f"mean={dataset_active_padded['active_mean']:.4f}, std={dataset_active_padded['active_std']:.4f}, "
            f"n_active={len(dataset_active_padded['active_dims'])}, "
            f"n_padded={len(dataset_active_padded['padded_dims'])}",
        )
        _check(
            "dataset padded dims are exactly zero",
            dataset_active_padded["padded_max_abs"] <= 1e-6,
            f"padded max |value|={dataset_active_padded['padded_max_abs']:.2e}",
        )

    if cache_full_stats is not None:
        cache_norm = (
            abs(cache_full_stats["global_mean"]) < NORMALIZED_GLOBAL_MEAN_TOL
            and NORMALIZED_GLOBAL_STD_LO < cache_full_stats["global_std"] < NORMALIZED_GLOBAL_STD_HI
        )
        _check(
            "cache global is normalized",
            cache_norm,
            f"mean={cache_full_stats['global_mean']:.4f}, std={cache_full_stats['global_std']:.4f}",
        )
    if cache_active_padded is not None:
        active_norm_c = (
            abs(cache_active_padded["active_mean"]) < NORMALIZED_GLOBAL_MEAN_TOL
            and NORMALIZED_GLOBAL_STD_LO < cache_active_padded["active_std"] < NORMALIZED_GLOBAL_STD_HI
        )
        _check(
            "cache active dims are normalized",
            active_norm_c,
            f"mean={cache_active_padded['active_mean']:.4f}, std={cache_active_padded['active_std']:.4f}, "
            f"n_active={len(cache_active_padded['active_dims'])}, "
            f"n_padded={len(cache_active_padded['padded_dims'])}",
        )
        _check(
            "cache padded dims are exactly zero",
            cache_active_padded["padded_max_abs"] <= 1e-6,
            f"padded max |value|={cache_active_padded['padded_max_abs']:.2e}",
        )

    if comparison is not None and not comparison.get("skipped"):
        _check(
            "cache vs dataset distributions match",
            bool(comparison["passes_tolerance"]),
            f"max_mean_diff={comparison['max_mean_diff']:.4f}, max_std_diff={comparison['max_std_diff']:.4f}",
        )

    if identity is not None and not identity.get("skipped"):
        _check(
            "cache identity matches live dataset",
            identity["failed"] == 0 and identity["checked"] > 0,
            f"checked={identity['checked']}, passed={identity['passed']}, failed={identity['failed']}",
        )

    if not checks:
        print("  (no checks ran -- see warnings above)")
        return

    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_total = len(checks)
    for name, ok, detail in checks:
        prefix = "[OK]  " if ok else "[WARN]"
        print(f"  {prefix} {name:<50}  {detail}")
    print()
    overall = n_pass == n_total
    print(f"  Overall: {n_pass}/{n_total} checks passed -- "
          f"{'HEALTHY' if overall else 'NEEDS REVIEW'}")
    if not overall:
        print("  Action: review the [WARN] lines above and the per-dim / per-subset tables.")


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_arguments()

    _section("ACTION DISTRIBUTION HEALTH CHECK")
    print(f"  cwd:                 {os.getcwd()}")
    print(f"  dataset_specs:       {args.dataset_specs}")
    print(f"  teacher_cache_path:  {args.teacher_cache_path or '(none)'}")
    print(f"  num_samples:         {args.num_samples}")
    print(f"  num_frames:          {args.num_frames}")
    print(f"  data_split:          {args.data_split}")
    print(f"  test_split_ratio:    {args.test_split_ratio}")
    sampling_mode = "random" if args.random else ("uniform" if args.uniform else "sequential")
    print(f"  sampling:            {sampling_mode}")
    print(f"  identity_check:      {args.identity_check_count}")
    print(f"  DEBUG_NORMALIZATION: {os.environ.get('DEBUG_NORMALIZATION', '0')}")

    specs, default_max_action_dim = load_dataset_specs(args.dataset_specs)
    max_action_dim = args.max_action_dim or default_max_action_dim
    print(f"  resolved max_action_dim: {max_action_dim}")
    print(f"  resolved {len(specs)} subset(s) from {args.dataset_specs}")

    dataset = None
    subset_info: list[dict] = []
    dataset_actions: Optional[np.ndarray] = None
    dataset_subsets: list[Optional[int]] = []
    chunk_size: int = -1

    if not args.skip_dataset:
        _section(f"BUILDING TRAINING-PIPELINE DATASET ({args.dataset_specs})")
        try:
            dataset = build_mixed_dataset(
                specs=specs,
                num_frames=args.num_frames,
                data_split=args.data_split,
                max_action_dim=max_action_dim,
                test_split_ratio=args.test_split_ratio,
            )
        except Exception as e:
            print(f"  FATAL: failed to build MixedLeRobotDataset: {e}")
            print("  Tip: re-run with --skip_dataset to analyze only the cache.")
            sys.exit(2)
        subset_info = get_subset_info(dataset)
        if subset_info:
            print("\n  Subset index ranges (virtual, post-repeat):")
            for s in subset_info:
                print(
                    f"    [{s['index']}] {s['name']:<32} {str(s['embodiment']):<18} "
                    f"real_len={s['real_len']:>9,}  virtual=[{s['virtual_idx_start']:>9,}, "
                    f"{s['virtual_idx_end_excl']:>9,})"
                )

        _section("SAMPLING TRAINING-PIPELINE ACTIONS")
        indices = make_indices(args.num_samples, len(dataset), sampling_mode, args.seed)
        print(
            f"  sampling {len(indices)} indices (mode={sampling_mode}): "
            f"min={int(indices.min())}, max={int(indices.max())}"
        )
        dataset_actions, dataset_subsets = collect_dataset_actions(dataset, indices, subset_info)
        chunk_size = dataset_actions.shape[0] // max(len(indices), 1)
        print(
            f"  collected {dataset_actions.shape[0]} action steps "
            f"({len(indices)} samples x ~{chunk_size} steps/sample)"
        )

    cache_actions: Optional[np.ndarray] = None
    cache_indices: list[int] = []
    cache_subsets: list[Optional[int]] = []
    if args.teacher_cache_path:
        _section(f"LOADING TEACHER CACHE  ({args.teacher_cache_path})")
        try:
            cache_actions, cache_indices, cache_subsets = collect_cache_actions(
                args.teacher_cache_path, args.num_samples, subset_info
            )
        except Exception as e:
            print(f"  FATAL: failed to load cache: {e}")
            cache_actions = None
        if cache_actions is not None:
            print(
                f"  collected {cache_actions.shape[0]} action steps "
                f"from {len(cache_indices)} cache files"
            )
            if cache_indices:
                print(
                    f"  cache index range:   min={min(cache_indices)},  max={max(cache_indices)}"
                )
                if subset_info:
                    counts: dict[Optional[int], int] = {}
                    for sid in cache_subsets:
                        counts[sid] = counts.get(sid, 0) + 1
                    print("  cache index -> subset distribution:")
                    for sid in sorted(
                        counts.keys(), key=lambda x: (x is None, x if x is not None else -1)
                    ):
                        if sid is None:
                            label = "<unmapped (out of dataset range)>"
                        elif 0 <= sid < len(subset_info):
                            label = f"[{sid}] {subset_info[sid]['name']}"
                        else:
                            label = f"sid={sid}"
                        n = counts[sid]
                        pct = 100.0 * n / len(cache_indices)
                        print(f"    {label:<48}  {n:>6}  {pct:>5.1f}%")

    dataset_full_stats = None
    cache_full_stats = None
    dataset_active_padded = None
    cache_active_padded = None

    if dataset_actions is not None:
        _section("TRAINING-PIPELINE STATS")
        dataset_full_stats = compute_full_stats(dataset_actions)
        print_global_block("dataset", dataset_full_stats)
        active_d, padded_d = detect_active_padded_dims(dataset_actions)
        dataset_active_padded = print_active_padded_breakdown(
            "dataset", dataset_actions, active_d, padded_d
        )
        if args.verbose:
            print_per_dim_table("dataset", dataset_actions, active_d)
        if subset_info and chunk_size > 0:
            print_per_subset(
                "dataset", dataset_actions, dataset_subsets, chunk_size, subset_info
            )

    if cache_actions is not None:
        _section("TEACHER CACHE STATS")
        cache_full_stats = compute_full_stats(cache_actions)
        print_global_block("cache", cache_full_stats)
        active_c, padded_c = detect_active_padded_dims(cache_actions)
        cache_active_padded = print_active_padded_breakdown(
            "cache", cache_actions, active_c, padded_c
        )
        if args.verbose:
            print_per_dim_table("cache", cache_actions, active_c)
        if subset_info and len(cache_indices) > 0:
            cache_chunk_size = cache_actions.shape[0] // len(cache_indices)
            print_per_subset(
                "cache", cache_actions, cache_subsets, cache_chunk_size, subset_info
            )

    comparison = None
    if dataset_actions is not None and cache_actions is not None:
        _section("CACHE vs DATASET COMPARISON")
        comparison = compare_distributions(
            "cache", cache_actions, "dataset", dataset_actions
        )

    identity = None
    if dataset is not None and args.teacher_cache_path and args.identity_check_count > 0:
        _section("IDENTITY CHECK")
        identity = identity_check(dataset, args.teacher_cache_path, args.identity_check_count)

    healthy_summary(
        cache_full_stats=cache_full_stats,
        dataset_full_stats=dataset_full_stats,
        cache_active_padded=cache_active_padded,
        dataset_active_padded=dataset_active_padded,
        comparison=comparison,
        identity=identity,
    )

    if args.save_report:
        report = {
            "args": vars(args),
            "subset_info": subset_info,
            "dataset_full_stats": dataset_full_stats,
            "cache_full_stats": cache_full_stats,
            "dataset_active_padded": dataset_active_padded,
            "cache_active_padded": cache_active_padded,
            "comparison": comparison,
            "identity": identity,
            "cache_index_range": {
                "min": int(min(cache_indices)) if cache_indices else None,
                "max": int(max(cache_indices)) if cache_indices else None,
                "n": len(cache_indices),
            },
        }
        out_path = Path(args.save_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(to_json_safe(report), f, indent=2)
        print(f"\n  saved JSON report to: {out_path}")

    print("\n  [done]")


if __name__ == "__main__":
    main()
