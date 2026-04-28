# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Validation quant-eval worker for in-training quality gating.

Runs as a SLURM-submitted subprocess from the
``EveryNValidationQuantEval`` callback at every N training steps.  It:

  1. Converts the latest DCP checkpoint shard for ``--checkpoint-iter`` to a
     single ``model_ema_bf16.pt`` (skips if already converted).
  2. Loads a small validation subset of the JHU dVRK mono finetune mixture
     (3 datasets x N episodes x M seeds by default), runs the existing
     two-phase quant evaluation (FDS + GATC + TCD), and saves the
     comparison videos under ``<validation_dir>/comparison/``.
  3. Generates a small OOD-scenario set on ``hf_suturebot`` (depth-only, 1
     episode by default) and saves the videos under ``<validation_dir>/ood/``.
  4. Resumes the training run's WandB session (via ``wandb_id.txt`` written
     by the trainer) and logs aggregated metrics + a few sample frames /
     mp4s under the ``val/...`` namespace.
  5. Appends a row to ``<validation_dir>/../metrics_history.csv`` so the
     progression across all validations of the run is one table on disk.

The callback fires this worker via ``sbatch`` (Option C: async SLURM job).
Training is not blocked: the eval lands in WandB minutes later.

Example invocation (the callback constructs this command for you)::

    PYTHONPATH=. python scripts/validation_quant_eval_worker.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss \\
        --checkpoint-dir /lustre/.../checkpoints/iter_000002000 \\
        --checkpoint-iter 2000 \\
        --validation-dir /workspace/validation/iter_000002000 \\
        --wandb-id-file /lustre/.../wandb_id.txt \\
        --metrics-history-csv /workspace/validation/metrics_history.csv \\
        --sam3-checkpoint /lustre/.../checkpoint_8_new_best.pt \\
        --val-datasets hf_suturebot cosmos_knot_fail_demo cosmos_fail_filtered \\
        --num-episodes 2 --num-seeds 2 \\
        --ood-datasets hf_suturebot --ood-episodes 1 --ood-depth-only

The worker does NOT block on its parent training job; it just exits after
posting metrics.  If anything fails, the failure is logged locally and
to WandB but training is unaffected.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

# Quant-eval reuses these public helpers from the existing scripts.  The two
# CLIs themselves are not invoked: we drive them programmatically.
from scripts.cosmos_h_surgical_simulator_quant_eval import (  # noqa: E402
    aggregate_checkpoint_results,
    evaluate_checkpoint,
    save_results_json,
)
from scripts.generate_open_h_ood_scenarios_depth import (  # noqa: E402
    ARM_LAYOUTS,
    NUM_OOD_CHUNKS,
    _auto_detect_arm_layout,
    compute_action_statistics,
    run_episode_ood,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (  # noqa: E402
    WrappedLeRobotSingleDataset,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import (  # noqa: E402
    EmbodimentTag,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (  # noqa: E402
    EMBODIMENT_REGISTRY,
    JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS,
    MAX_ACTION_DIM,
    construct_modality_config_and_transforms,
)
from cosmos_predict2._src.predict2.action.inference.inference_open_h import (  # noqa: E402
    CHUNK_SIZE,
    build_episode_index_map,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (  # noqa: E402
    ActionVideo2WorldInference,
)


# =============================================================================
# Constants
# =============================================================================

# Default validation subset (3 datasets, kept small to fit in ~30 min wall time).
# Names must match ``Path.name`` of entries in ``JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS``.
DEFAULT_VAL_DATASETS = ("hf_suturebot", "cosmos_knot_fail_demo", "cosmos_fail_filtered")
DEFAULT_OOD_DATASETS = ("hf_suturebot",)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation quant-eval worker (run by the EveryNValidationQuantEval callback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Identifiers ---
    p.add_argument("--experiment", required=True,
                   help="Training experiment name (e.g. cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss)")
    p.add_argument("--checkpoint-iter", type=int, required=True,
                   help="Iteration number this validation pertains to (e.g. 2000).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to the DCP checkpoint directory for this iter (must contain a 'model/' subdir).")
    p.add_argument("--validation-dir", required=True,
                   help="Output directory for this validation (e.g. <repo_root>/validation/iter_000002000).")
    p.add_argument("--wandb-id-file", required=True,
                   help="Path to wandb_id.txt written by the trainer at start of training.")
    p.add_argument("--metrics-history-csv", required=True,
                   help="Append-only CSV that accumulates per-iteration aggregated metrics across the run.")

    # --- SAM3 (required for GATC + TCD) ---
    p.add_argument("--sam3-checkpoint", required=True,
                   help="Path to the Medical-SAM3 .pt checkpoint.")
    p.add_argument("--sam3-prompt", default="surgical tool")
    p.add_argument("--sam3-score-threshold", type=float, default=0.3)

    # --- Validation scope ---
    p.add_argument("--val-datasets", nargs="+", default=list(DEFAULT_VAL_DATASETS),
                   help="Subset of validation datasets (matched by basename of spec path).")
    p.add_argument("--num-episodes", type=int, default=2)
    p.add_argument("--num-seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--data-split", default="test")
    p.add_argument("--gatc-k", type=int, default=3)
    p.add_argument("--gatc-use-grad", action="store_true")

    # --- OOD scope ---
    p.add_argument("--ood-datasets", nargs="+", default=list(DEFAULT_OOD_DATASETS),
                   help="Subset of datasets to generate OOD videos for (basename match).")
    p.add_argument("--ood-episodes", type=int, default=1)
    p.add_argument("--ood-depth-only", action="store_true", default=True,
                   help="Generate only the 2 depth scenarios (push/pull) per episode. Default: true.")
    p.add_argument("--ood-magnitude", type=float, default=1.0)
    p.add_argument("--ood-stats-episodes", type=int, default=10)

    # --- Conversion / model loading ---
    p.add_argument("--s3-cred", default="credentials/s3_checkpoint.secret")
    p.add_argument("--save-fps", type=int, default=10)
    p.add_argument("--skip-conversion", action="store_true",
                   help="Assume model_ema_bf16.pt already exists in checkpoint-dir; just run eval.")

    return p.parse_args()


# =============================================================================
# DCP -> .pt conversion
# =============================================================================

def convert_dcp_to_pt(checkpoint_dir: str, ema: bool = True) -> str:
    """Convert a sharded DCP checkpoint to a single ``.pt`` file.

    Mirrors the logic of ``scripts/convert_distcp_to_pt.py`` but as an in-process
    function so the worker doesn't need to fork to a subprocess.

    Args:
        checkpoint_dir: Directory containing the DCP shards (must have a ``model/`` subdir).
        ema: If True, also export EMA-only weights in fp32 and bf16.

    Returns:
        Absolute path to ``model_ema_bf16.pt`` (or ``model.pt`` if ``ema=False``).
    """
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    out_dir = Path(checkpoint_dir)
    distcp_dir = out_dir / "model"
    if not distcp_dir.is_dir():
        raise FileNotFoundError(f"Expected DCP shard dir at {distcp_dir}")

    pt_path = out_dir / "model.pt"
    pt_ema_fp32_path = out_dir / "model_ema_fp32.pt"
    pt_ema_bf16_path = out_dir / "model_ema_bf16.pt"

    # Idempotent: if already converted, skip.
    if ema and pt_ema_bf16_path.is_file():
        logger.info(f"DCP->pt: {pt_ema_bf16_path.name} already exists, reusing.")
        return str(pt_ema_bf16_path)
    if not ema and pt_path.is_file():
        return str(pt_path)

    pt_path.unlink(missing_ok=True)
    pt_ema_fp32_path.unlink(missing_ok=True)
    pt_ema_bf16_path.unlink(missing_ok=True)

    logger.info(f"DCP->pt: converting {distcp_dir} -> {pt_path}")
    t0 = time.time()
    dcp_to_torch_save(str(distcp_dir), str(pt_path))
    logger.info(f"DCP->pt: dcp_to_torch_save done in {time.time() - t0:.1f}s")

    if not ema:
        return str(pt_path)

    state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    state_dict_ema_fp32: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("net_ema."):
            state_dict_ema_fp32[key.replace("net_ema.", "net.")] = value
    if not state_dict_ema_fp32:
        raise ValueError(f"Checkpoint {pt_path} contains no EMA weights (no 'net_ema.' keys).")
    torch.save(state_dict_ema_fp32, str(pt_ema_fp32_path))

    state_dict_ema_bf16: Dict[str, Any] = {}
    for key, value in state_dict_ema_fp32.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            value = value.bfloat16()
        state_dict_ema_bf16[key] = value
    torch.save(state_dict_ema_bf16, str(pt_ema_bf16_path))

    # Drop the verbose intermediates to save disk; bf16 is what inference loads.
    pt_path.unlink(missing_ok=True)
    pt_ema_fp32_path.unlink(missing_ok=True)
    logger.info(f"DCP->pt: wrote EMA bf16 at {pt_ema_bf16_path}")
    return str(pt_ema_bf16_path)


# =============================================================================
# Validation dataset selection
# =============================================================================

def _select_val_dataset_specs(
    requested_names: Sequence[str],
) -> List[dict]:
    """Filter ``JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS`` to the requested basenames.

    Keeps the original spec dict (so ``embodiment``, ``test_split_ratio_override``,
    etc. are preserved) but only returns entries whose ``Path(spec["path"]).name``
    is in ``requested_names``.
    """
    by_name = {Path(spec["path"]).name: spec for spec in JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS}
    selected: List[dict] = []
    missing: List[str] = []
    for name in requested_names:
        if name in by_name:
            selected.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        logger.warning(
            f"Requested validation datasets not found in JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS: {missing}. "
            f"Available: {sorted(by_name.keys())}"
        )
    return selected


def _build_evaluation_dataset_configs(
    specs: Sequence[dict],
    num_episodes: int,
    base_seed: int,
    data_split: str,
) -> List[dict]:
    """For each val spec, enumerate test-split episode IDs and pick ``num_episodes`` of them.

    Uses ``WrappedLeRobotSingleDataset(data_split="test")`` to enumerate which
    episodes are present in the held-out partition (the same partition the
    training trainer's ``data_val`` uses), then picks the first ``num_episodes``
    that have a valid ``base_index=0`` chunk-start.

    Returns a list of dataset_configs in the format ``evaluate_checkpoint``
    expects (matches the multi-dataset mode of cosmos_h_surgical_simulator_quant_eval.py).
    """
    configs: List[dict] = []
    for spec in specs:
        dataset_path = spec["path"]
        embodiment_tag = spec["embodiment"]
        embodiment_str = embodiment_tag.value if isinstance(embodiment_tag, EmbodimentTag) else str(embodiment_tag)
        timestep_interval = EMBODIMENT_REGISTRY.get(embodiment_str, {}).get("timestep_interval", 6)

        # Load with test split to get the actual episodes the trainer uses for val
        config, _, test_transform = construct_modality_config_and_transforms(
            num_frames=13, embodiment=embodiment_str, downscaled_res=False,
        )
        modality_filename = None
        if isinstance(config, dict) and "modality_filename" in config:
            modality_filename = config.pop("modality_filename")
        ds = WrappedLeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=config,
            transforms=test_transform,
            embodiment_tag=embodiment_str,
            data_split=data_split,
            modality_filename=modality_filename,
        )
        episode_map = build_episode_index_map(ds)

        # Filter to episodes that have base_index=0 in the test split
        usable = sorted(
            ep for ep, entries in episode_map.items()
            if any(bi == 0 for _, bi in entries)
        )
        if not usable:
            logger.warning(f"[{Path(dataset_path).name}] No usable test-split episodes (no base_index=0). Skipping.")
            continue

        # Deterministic selection: first num_episodes after a stable shuffle by base_seed
        # (so different validations see different episode subsets if desired).
        rng = np.random.default_rng(base_seed)
        usable_arr = np.array(usable, dtype=np.int64)
        rng.shuffle(usable_arr)
        picked = sorted(usable_arr[: num_episodes].tolist())

        configs.append({
            "path": dataset_path,
            "name": Path(dataset_path).name,
            "episode_ids": picked,
            "embodiment": embodiment_str,
            "timestep_interval": int(timestep_interval),
            "exclude_splits": spec.get("exclude_splits", None),
        })
        logger.info(
            f"[{Path(dataset_path).name}] embodiment={embodiment_str}, timestep_interval={timestep_interval}, "
            f"episodes={picked} ({len(picked)}/{len(usable)} usable)"
        )

    return configs


# =============================================================================
# OOD generation (small subset)
# =============================================================================

def _resolve_arm_layout(embodiment: str, raw_dim: int) -> dict:
    layout = ARM_LAYOUTS.get(embodiment)
    if layout is None:
        layout = _auto_detect_arm_layout(embodiment, raw_dim)
    return layout


def _run_ood_for_dataset(
    *,
    video2world: ActionVideo2WorldInference,
    spec: dict,
    save_dir: str,
    num_episodes: int,
    seed: int,
    guidance: float,
    save_fps: int,
    ood_magnitude: float,
    stats_episodes: int,
    depth_only: bool,
) -> Tuple[int, List[dict]]:
    """Generate OOD scenario videos for one dataset.

    Returns
    -------
    n_scenarios_total : int
        Total number of OOD videos generated (sum across episodes).
    episode_meta : list[dict]
        Per-episode metadata: ``{"episode_id", "scenarios": [{"filename", ...}, ...]}``.
    """
    dataset_path = spec["path"]
    embodiment_tag = spec["embodiment"]
    embodiment_str = embodiment_tag.value if isinstance(embodiment_tag, EmbodimentTag) else str(embodiment_tag)
    timestep_interval = EMBODIMENT_REGISTRY.get(embodiment_str, {}).get("timestep_interval", 6)

    config, _, test_transform = construct_modality_config_and_transforms(
        num_frames=13, embodiment=embodiment_str, downscaled_res=False,
    )
    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")
    ds = WrappedLeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=config,
        transforms=test_transform,
        embodiment_tag=embodiment_str,
        data_split="test",
        modality_filename=modality_filename,
    )
    episode_map = build_episode_index_map(ds)

    # Need enough chunks for OOD rollout (NUM_OOD_CHUNKS) plus initial frame
    timestep_stride = CHUNK_SIZE * timestep_interval
    min_chunks_needed = max(NUM_OOD_CHUNKS, 2)
    max_base_index = (min_chunks_needed - 1) * timestep_stride
    usable = [
        ep for ep in episode_map
        if any(bi == 0 for _, bi in episode_map[ep])
        and any(bi >= max_base_index for _, bi in episode_map[ep])
    ]
    if not usable:
        logger.warning(f"OOD [{Path(dataset_path).name}]: no usable episodes (>= {min_chunks_needed} chunks).")
        return 0, []

    rng = np.random.default_rng(seed)
    usable_arr = np.array(sorted(usable), dtype=np.int64)
    rng.shuffle(usable_arr)
    picked = sorted(usable_arr[: num_episodes].tolist())

    # Compute action stats once, then call run_episode_ood per episode
    mean_action, std_action, raw_dim, stacked = compute_action_statistics(
        ds, episode_map, timestep_interval, max_episodes=stats_episodes, seed=seed,
    )
    # Frequent actions are not used in depth-only mode; keep an empty list to satisfy run_episode_ood signature.
    frequent_actions: List[Tuple[np.ndarray, int]] = []
    arm_layout = _resolve_arm_layout(embodiment_str, raw_dim)

    episode_meta: List[dict] = []
    n_total = 0
    for ep in picked:
        try:
            n_ood = run_episode_ood(
                video2world=video2world,
                dataset=ds,
                episode_map=episode_map,
                episode_id=ep,
                timestep_interval=timestep_interval,
                arm_layout=arm_layout,
                mean_action=mean_action,
                std_action=std_action,
                raw_dim=raw_dim,
                frequent_actions=frequent_actions,
                total_action_steps=len(stacked),
                save_dir=save_dir,
                seed=seed,
                guidance=guidance,
                save_fps=save_fps,
                ood_magnitude=ood_magnitude,
                depth_only=depth_only,
            )
        except Exception:
            logger.error(f"OOD [{Path(dataset_path).name}] ep={ep} failed:\n{traceback.format_exc()}")
            n_ood = 0
        if n_ood > 0:
            n_total += n_ood
            episode_meta.append({"episode_id": int(ep), "n_scenarios": int(n_ood)})

    return n_total, episode_meta


# =============================================================================
# WandB logging
# =============================================================================

def _read_wandb_id(wandb_id_file: str) -> Optional[str]:
    if not os.path.isfile(wandb_id_file):
        logger.warning(f"wandb_id file not found at {wandb_id_file} — WandB logging disabled.")
        return None
    with open(wandb_id_file, "r") as f:
        wid = f.read().strip()
    return wid or None


def _log_to_wandb(
    *,
    wandb_id: Optional[str],
    experiment: str,
    iteration: int,
    agg: dict,
    validation_dir: str,
    save_fps: int,
) -> None:
    """Resume the training run's WandB session and log val/* metrics + a couple of mp4s.

    Best-effort: any logging error is logged locally and swallowed (the eval
    JSON / CSV on disk is the canonical record).
    """
    if wandb_id is None:
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not importable; skipping WandB logging.")
        return

    # Minimal headline scalars only.  Per-dataset, per-chunk (early/mid/late),
    # SSIM, mean (vs. median), coverage and slope variants are still written to
    # ``metrics.json`` and ``metrics_history.csv`` on disk -- we just don't push
    # them to WandB to keep the run dashboard scannable.  Less is more.
    info: Dict[str, Any] = {"trainer/global_step": iteration}
    info["val/fds/l1_mean"] = agg.get("l1_mean", float("nan"))
    if "gatc_median" in agg:
        info["val/gatc/median"] = agg["gatc_median"]
    if "tcd_median" in agg:
        info["val/tcd/median_px"] = agg["tcd_median"]

    # Resume the same run as the trainer.
    try:
        wandb.init(
            id=wandb_id,
            resume="must",
            project="cosmos_predict2_action_conditioned",
            name=experiment,
        )
    except Exception:
        logger.error(f"wandb.init(resume='must') failed for id={wandb_id}:\n{traceback.format_exc()}")
        return

    # Attach a few mp4s (one per dataset, one OOD)
    comparison_dir = Path(validation_dir) / "comparison"
    if comparison_dir.is_dir():
        for ds_dir in comparison_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            mp4s = sorted(ds_dir.glob("*.mp4"))
            if mp4s:
                info[f"val/comparison_videos/{ds_dir.name}"] = wandb.Video(str(mp4s[0]), fps=save_fps, format="mp4")

    ood_dir = Path(validation_dir) / "ood"
    if ood_dir.is_dir():
        # Find any single OOD scenario mp4 to attach as a sanity check
        mp4s = sorted(ood_dir.rglob("*.mp4"))
        for mp4 in mp4s[:2]:
            scenario = mp4.stem
            info[f"val/ood/{scenario}"] = wandb.Video(str(mp4), fps=save_fps, format="mp4")

    try:
        wandb.log(info, step=iteration)
        logger.info(f"WandB: posted {len(info)} val/* metrics + videos for iter {iteration}")
    finally:
        wandb.finish(quiet=True)


# =============================================================================
# Local persistence
# =============================================================================

def _append_metrics_history_csv(csv_path: str, iteration: int, agg: dict) -> None:
    """Append a single row to the run's metrics_history.csv (creates header on first write)."""
    fields = [
        "iteration", "timestamp",
        "fds_l1_mean", "fds_ssim_mean", "fds_l1_slope_mean",
        "fds_l1_early", "fds_l1_mid", "fds_l1_late",
        "gatc_median", "gatc_coverage_pct",
        "gatc_early_median", "gatc_mid_median", "gatc_late_median",
        "tcd_median_px", "tcd_coverage_pct",
        "tcd_early_median_px", "tcd_mid_median_px", "tcd_late_median_px",
    ]
    row = {
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fds_l1_mean": agg.get("l1_mean", float("nan")),
        "fds_ssim_mean": agg.get("ssim_mean", float("nan")),
        "fds_l1_slope_mean": agg.get("l1_slope_mean", float("nan")),
        "fds_l1_early": agg.get("l1_early_c1", float("nan")),
        "fds_l1_mid": agg.get("l1_mid_c2c3", float("nan")),
        "fds_l1_late": agg.get("l1_late_c4c6", float("nan")),
        "gatc_median": agg.get("gatc_median", float("nan")),
        "gatc_coverage_pct": agg.get("gatc_coverage_mean", float("nan")),
        "gatc_early_median": agg.get("gatc_early_c1_median", float("nan")),
        "gatc_mid_median": agg.get("gatc_mid_c2c3_median", float("nan")),
        "gatc_late_median": agg.get("gatc_late_c4c6_median", float("nan")),
        "tcd_median_px": agg.get("tcd_median", float("nan")),
        "tcd_coverage_pct": agg.get("tcd_coverage_mean", float("nan")),
        "tcd_early_median_px": agg.get("tcd_early_c1_median", float("nan")),
        "tcd_mid_median_px": agg.get("tcd_mid_c2c3_median", float("nan")),
        "tcd_late_median_px": agg.get("tcd_late_c4c6_median", float("nan")),
    }
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    is_new = not Path(csv_path).is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _flatten_eval_video_layout(validation_dir: str, label: str) -> None:
    """Flatten the auto-created ``<validation_dir>/<label>/<dataset>/`` layout.

    ``evaluate_checkpoint(..., save_videos=True)`` writes
    ``<save_path>/<label>/<dataset>/{generated,comparison}/<tag>.mp4``.  We
    want the comparison videos directly under ``<validation_dir>/comparison/<dataset>/``
    so the validation directory is one consistent shape.  We move only the
    comparison side-by-sides (drop the generated-only copies, since the
    side-by-side already shows generation alongside GT).
    """
    src_root = Path(validation_dir) / label
    if not src_root.is_dir():
        return
    dst_root = Path(validation_dir) / "comparison"
    dst_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    for ds_dir in src_root.iterdir():
        if not ds_dir.is_dir():
            continue
        cmp_dir = ds_dir / "comparison"
        if not cmp_dir.is_dir():
            continue
        target_ds = dst_root / ds_dir.name
        target_ds.mkdir(parents=True, exist_ok=True)
        for mp4 in cmp_dir.glob("*.mp4"):
            mp4.rename(target_ds / mp4.name)
            moved += 1
    # Best-effort cleanup of the now-empty nested tree (don't fail if it has
    # other entries we don't recognise).
    try:
        for ds_dir in list(src_root.iterdir()):
            for sub in list(ds_dir.iterdir()):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            if ds_dir.is_dir() and not any(ds_dir.iterdir()):
                ds_dir.rmdir()
        if not any(src_root.iterdir()):
            src_root.rmdir()
    except OSError:
        pass
    logger.info(f"Flattened {moved} comparison videos into {dst_root}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)

    Path(args.validation_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.validation_dir) / "worker.log"
    logger.add(str(log_path), level="INFO", enqueue=False, backtrace=False, diagnose=False)

    logger.info("=" * 80)
    logger.info(f"Validation worker starting | iter={args.checkpoint_iter}")
    logger.info(f"  experiment      = {args.experiment}")
    logger.info(f"  checkpoint_dir  = {args.checkpoint_dir}")
    logger.info(f"  validation_dir  = {args.validation_dir}")
    logger.info(f"  val_datasets    = {args.val_datasets}")
    logger.info(f"  ood_datasets    = {args.ood_datasets}")
    logger.info(f"  num_episodes    = {args.num_episodes}, num_seeds = {args.num_seeds}")
    logger.info("=" * 80)

    wandb_id = _read_wandb_id(args.wandb_id_file)

    # ------------------------------------------------------------------
    # 1. DCP -> .pt
    # ------------------------------------------------------------------
    if args.skip_conversion:
        pt_path = str(Path(args.checkpoint_dir) / "model_ema_bf16.pt")
    else:
        try:
            pt_path = convert_dcp_to_pt(args.checkpoint_dir, ema=True)
        except Exception:
            logger.error(f"DCP->pt conversion failed:\n{traceback.format_exc()}")
            return 1

    # ------------------------------------------------------------------
    # 2. Build evaluation dataset configs from the val mixture
    # ------------------------------------------------------------------
    specs = _select_val_dataset_specs(args.val_datasets)
    if not specs:
        logger.error("No matching val datasets found, aborting.")
        return 2

    eval_configs = _build_evaluation_dataset_configs(
        specs=specs,
        num_episodes=args.num_episodes,
        base_seed=args.seed + args.checkpoint_iter,  # different selection per iter
        data_split=args.data_split,
    )
    if not eval_configs:
        logger.error("No evaluable episodes in any selected val dataset, aborting.")
        return 3

    # ------------------------------------------------------------------
    # 3. Run quant eval (Phase 1 + Phase 2)
    # ------------------------------------------------------------------
    seeds = [args.seed + i for i in range(args.num_seeds)]
    eval_args = argparse.Namespace(
        experiment=args.experiment,
        s3_cred=args.s3_cred,
        context_parallel_size=1,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_prompt=args.sam3_prompt,
        sam3_score_threshold=args.sam3_score_threshold,
        gatc_k=args.gatc_k,
        gatc_use_grad=args.gatc_use_grad,
        guidance=args.guidance,
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_episodes=args.num_episodes,
        episodes_per_dataset=args.num_episodes,
        save_path=args.validation_dir,
        save_fps=args.save_fps,
        save_videos=True,            # writes to <validation_dir>/<label>/<ds>/{generated,comparison}/
        data_split=args.data_split,
        test_episodes_json=None,     # not used since we pass dataset_configs directly
        exclude_datasets=[],
        include_datasets=[],
    )

    label = f"iter_{args.checkpoint_iter:09d}"
    t0 = time.time()
    try:
        episodes = evaluate_checkpoint(eval_args, pt_path, label, seeds, dataset_configs=eval_configs)
    except Exception:
        logger.error(f"evaluate_checkpoint failed:\n{traceback.format_exc()}")
        return 4

    if not episodes:
        logger.error("evaluate_checkpoint returned no episodes, aborting.")
        return 5

    # evaluate_checkpoint already wrote comparison + generated mp4s under
    # <validation_dir>/<label>/<ds>/{comparison,generated}/.  Flatten the
    # comparison side-by-sides into <validation_dir>/comparison/<ds>/ for a
    # cleaner layout, and drop the generated-only duplicates.
    _flatten_eval_video_layout(args.validation_dir, label)

    agg = aggregate_checkpoint_results(episodes)
    save_results_json(
        [(label, pt_path, agg, episodes)], seeds, eval_args, args.validation_dir,
        dataset_configs=eval_configs,
    )

    quant_eval_dur = time.time() - t0
    logger.info(f"Quant eval done in {quant_eval_dur / 60:.1f} min — FDS L1={agg.get('l1_mean', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 4. OOD generation on the small ood_datasets subset
    # ------------------------------------------------------------------
    ood_dir = Path(args.validation_dir) / "ood"
    ood_dir.mkdir(parents=True, exist_ok=True)
    ood_specs = _select_val_dataset_specs(args.ood_datasets)
    n_ood_total = 0
    ood_meta: Dict[str, List[dict]] = {}
    if ood_specs:
        try:
            video2world = ActionVideo2WorldInference(
                args.experiment, pt_path, args.s3_cred, context_parallel_size=1,
            )
            for spec in ood_specs:
                ds_save = str(ood_dir / Path(spec["path"]).name)
                Path(ds_save).mkdir(parents=True, exist_ok=True)
                n_total, ep_meta = _run_ood_for_dataset(
                    video2world=video2world,
                    spec=spec,
                    save_dir=ds_save,
                    num_episodes=args.ood_episodes,
                    seed=args.seed,
                    guidance=args.guidance,
                    save_fps=args.save_fps,
                    ood_magnitude=args.ood_magnitude,
                    stats_episodes=args.ood_stats_episodes,
                    depth_only=args.ood_depth_only,
                )
                n_ood_total += n_total
                ood_meta[Path(spec["path"]).name] = ep_meta
            video2world.cleanup()
            del video2world
            torch.cuda.empty_cache()
        except Exception:
            logger.error(f"OOD generation failed:\n{traceback.format_exc()}")

    logger.info(f"OOD generation done — {n_ood_total} videos across {len(ood_meta)} dataset(s)")

    # ------------------------------------------------------------------
    # 5. Persist + log
    # ------------------------------------------------------------------
    _append_metrics_history_csv(args.metrics_history_csv, args.checkpoint_iter, agg)

    metrics_path = Path(args.validation_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "iteration": args.checkpoint_iter,
            "experiment": args.experiment,
            "checkpoint": pt_path,
            "aggregated": agg,
            "ood": ood_meta,
            "scope": {
                "val_datasets": list(args.val_datasets),
                "ood_datasets": list(args.ood_datasets),
                "num_episodes": args.num_episodes,
                "num_seeds": args.num_seeds,
                "ood_episodes": args.ood_episodes,
                "ood_depth_only": args.ood_depth_only,
            },
        }, f, indent=2, default=str)
    logger.info(f"Wrote metrics summary to {metrics_path}")

    _log_to_wandb(
        wandb_id=wandb_id,
        experiment=args.experiment,
        iteration=args.checkpoint_iter,
        agg=agg,
        validation_dir=args.validation_dir,
        save_fps=args.save_fps,
    )

    logger.info(f"Validation worker done | iter={args.checkpoint_iter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
