#!/usr/bin/env python
"""
Generate out-of-distribution (OOD) action trajectory scenario videos for Open-H.

For each dataset in a test_episodes.json file, this script:
1. Saves the ground-truth video (6 chunks = 72 predicted frames + 1 conditioning).
2. Generates the model's predicted video using ground-truth actions (same framing).
3. Creates up to 14 OOD scenario videos (5 chunks = 60 predicted frames + 1 conditioning,
   i.e. 5 seconds at 12 fps) using synthetic action trajectories that move arms in
   specific directions or repeat frequently occurring actions from the dataset.

The OOD trajectories are constructed by:
- Computing action statistics (mean, std) from sampled episodes per dataset
- Identifying which action dimensions control horizontal (x) and vertical (y)
  movement for each arm via pre-defined arm layout mappings
- Setting those dimensions to ±k standard deviations from the mean while keeping
  all other dimensions at neutral (mean) values
- Clustering sampled actions via k-means to find the 3 most frequently occurring
  action patterns, then repeating each 60 times

The 14 OOD scenarios (or 2 with --depth_only):
  1)  Both arms left 60x
  2)  Both arms right 60x
  3)  Both arms up 60x
  4)  Both arms down 60x
  5)  Both arms left 30x, then both arms up 30x
  6)  Left arm left + right arm right 30x, then left up + right down 30x
  7)  Left arm up 60x, right arm down 60x
  8)  Left arm up 60x, right arm down 60x (duplicate of 7, per spec)
  9)  Left up + right down 30x, then left right + right left 30x
  10) Most frequent action pattern repeated 60x
  11) 2nd most frequent action pattern repeated 60x
  12) 3rd most frequent action pattern repeated 60x
  13) Both arms push into tissue (+z depth) 60x
  14) Both arms pull away from tissue (-z depth) 60x

Output structure:
    <save_root>/<timestamp>/<dataset_name>/
    ├── gt/episode_XXXX.mp4
    ├── gt_60pred/episode_XXXX.mp4
    ├── predicted/episode_XXXX.mp4
    ├── predicted_60pred/episode_XXXX.mp4
    └── ood_scenarios/episode_XXXX/
        ├── 01_both_left.mp4
        ├── 02_both_right.mp4
        ├── 03_both_up.mp4
        ├── 04_both_down.mp4
        ├── 05_both_left_then_up.mp4
        ├── 06_mirror_lr_then_ud.mp4
        ├── 07_left_up_right_down.mp4
        ├── 08_left_up_right_down_v2.mp4
        ├── 09_split_ud_then_rl.mp4
        ├── 10_frequent_pattern_1.mp4
        ├── 11_frequent_pattern_2.mp4
        ├── 12_frequent_pattern_3.mp4
        ├── 13_depth_push_into.mp4
        ├── 14_depth_pull_away.mp4
        └── scenarios.json

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/generate_open_h_ood_scenarios.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \\
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \\
        --test_episodes_json output/open-h_test_episodes.json \\
        --save_root results/open_h_ood \\
        --episodes_per_dataset 1 \\
        --seed 42
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)
from cosmos_predict2._src.predict2.action.inference.inference_open_h import (
    load_wrapped_dataset,
    resolve_timestep_interval,
    build_episode_index_map,
    find_chunk_indices,
    pad_action,
    _lookup_exclude_splits,
    CHUNK_SIZE,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    MAX_ACTION_DIM,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_GT_CHUNKS = 6                                  # 6 × 12 = 72 predicted frames
NUM_OOD_CHUNKS = 5                                 # 5 × 12 = 60 predicted frames (~5s at 12fps)
OOD_TRAJECTORY_LENGTH = NUM_OOD_CHUNKS * CHUNK_SIZE  # 60 action steps
SHORT_PRED_FRAMES = OOD_TRAJECTORY_LENGTH             # match OOD length for combining
SHORT_TOTAL_FRAMES = SHORT_PRED_FRAMES + 1            # includes initial conditioning frame


# ---------------------------------------------------------------------------
# Arm layout mapping
# ---------------------------------------------------------------------------
# Maps each embodiment tag to the indices of x (horizontal), y (vertical),
# and z (depth) position components for each arm in the concatenated un-padded
# action vector.
#
# For rel_xyz_rot6d pose keys the output is 9D (xyz_3D + rot6d_6D);
# index 0 = x (horizontal in image plane), index 1 = y (vertical),
# index 2 = z (depth — positive = into tissue, negative = away).
#
# "left" refers to the first arm in concatenation order, "right" the second.
# Single-arm embodiments have right_* = None and dual = False.
# None entry means the layout must be auto-detected at runtime (joint-space).
# ---------------------------------------------------------------------------

ARM_LAYOUTS: dict[str, dict | None] = {
    # --- Dual-arm with grippers: [pose(9)+grip(1)] × 2 = 20D ---
    "dvrk":               {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "jhu_dvrk_mono":      {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "dvrk_ucb":           {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "dvrk_obuda":         {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "dvrk_stanford_real": {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "jhu_lscr_miracle":   {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "jhu_lscr_smarts":    {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "hamlyn_30hz":        {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    "dvrk_ucsd":          {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
    # --- Dual-arm pose only (no grippers): [pose(9)] × 2 = 18D ---
    "turin_mitic_ex_vivo": {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 9, "right_y": 10, "right_z": 11, "dual": True},
    # --- 4-arm pose only: [pose(9)] × 4 = 36D; primary two arms ---
    "rob_surgical":       {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 9, "right_y": 10, "right_z": 11, "dual": True},
    # --- Single-arm with gripper: [pose(9)+grip(1)] = 10D ---
    "polyu_sim":          {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": None, "right_y": None, "right_z": None, "dual": False},
    "tud_tundra":         {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": None, "right_y": None, "right_z": None, "dual": False},
    # --- Dual-arm delta xyz: [right_xyz(3)+left_xyz(3)] = 6D (right first!) ---
    "moon":               {"left_x": 3, "left_y": 4, "left_z": 5, "right_x": 0, "right_y": 1, "right_z": 2, "dual": True},
    # --- Joint-space: auto-detected at runtime ---
    "ustc_torin":         None,
    # --- CMR Versius: [left_pose(9)+grip(1)+right_pose(9)+grip(1)+...] = 44D ---
    "cmr_versius":        {"left_x": 0, "left_y": 1, "left_z": 2, "right_x": 10, "right_y": 11, "right_z": 12, "dual": True},
}


def _auto_detect_arm_layout(embodiment: str, raw_action_dim: int) -> dict:
    """Infer arm layout for embodiments not pre-defined in ARM_LAYOUTS.

    For joint-space embodiments (e.g., USTC Torin) the per-dimension mapping to
    image-plane x/y/z is approximate — the first three dimensions of each arm's
    joint block are used as proxies for horizontal, vertical, and depth movement.

    Args:
        embodiment: Embodiment tag string.
        raw_action_dim: Total dimensionality of the concatenated un-padded action.

    Returns:
        Layout dict with left_x, left_y, left_z, right_x, right_y, right_z, dual keys.
    """
    if embodiment == "ustc_torin":
        half = raw_action_dim // 2
        logger.info(
            f"Auto-detected USTC Torin layout: {raw_action_dim}D total, "
            f"left=[0:{half}], right=[{half}:{raw_action_dim}]"
        )
        return {
            "left_x": 0, "left_y": 1, "left_z": 2,
            "right_x": half, "right_y": half + 1, "right_z": half + 2,
            "dual": True,
        }

    logger.warning(
        f"Unknown embodiment '{embodiment}' with raw_dim={raw_action_dim}: "
        "falling back to single-arm (first 3 dims as x/y/z)"
    )
    return {
        "left_x": 0, "left_y": 1, "left_z": 2,
        "right_x": None, "right_y": None, "right_z": None,
        "dual": False,
    }


# ---------------------------------------------------------------------------
# OOD scenario construction
# ---------------------------------------------------------------------------

def _build_ood_scenarios(
    arm_layout: dict,
    mean_action: np.ndarray,
    std_action: np.ndarray,
    magnitude: float = 1.0,
) -> list[tuple[str, str, np.ndarray]]:
    """Build all 9 OOD trajectory sequences from directional perturbations.

    Each scenario produces an ``OOD_TRAJECTORY_LENGTH``-step action trajectory
    (currently 60 steps = 5 chunks of 12) by perturbing the mean action by
    ±magnitude×std in the x/y position dimensions.  Non-position dimensions
    (rotation, gripper, conditioning) stay at their mean (neutral) values,
    representing no change.

    For single-arm embodiments, right-arm perturbations are silently ignored.

    Args:
        arm_layout: Dict with ``left_x``, ``left_y``, ``right_x``, ``right_y``,
            and ``dual`` flag.
        mean_action: Per-dimension mean of transformed actions, shape ``(D,)``.
        std_action: Per-dimension std, shape ``(D,)``.
        magnitude: Scaling factor for the directional perturbation in units of
            standard deviations (default 1.0 = one std).

    Returns:
        List of ``(filename, description, trajectory)`` tuples where *trajectory*
        has shape ``(OOD_TRAJECTORY_LENGTH, D)`` in the un-padded action space.
    """
    half = OOD_TRAJECTORY_LENGTH // 2
    lx, ly = arm_layout["left_x"], arm_layout["left_y"]
    rx, ry = arm_layout.get("right_x"), arm_layout.get("right_y")
    is_dual = arm_layout["dual"]

    def _make_action(left_dx=0.0, left_dy=0.0, right_dx=0.0, right_dy=0.0):
        """Single action step with directional perturbations (in std units)."""
        a = mean_action.copy()
        if lx is not None:
            a[lx] += left_dx * magnitude * std_action[lx]
        if ly is not None:
            a[ly] += left_dy * magnitude * std_action[ly]
        if is_dual and rx is not None:
            a[rx] += right_dx * magnitude * std_action[rx]
        if is_dual and ry is not None:
            a[ry] += right_dy * magnitude * std_action[ry]
        return a

    def _tile(action):
        return np.tile(action, (OOD_TRAJECTORY_LENGTH, 1))

    def _concat_halves(first, second):
        return np.concatenate([
            np.tile(first, (half, 1)),
            np.tile(second, (half, 1)),
        ], axis=0)

    full = OOD_TRAJECTORY_LENGTH  # 60
    return [
        # 1) Both arms go left
        ("01_both_left",
         f"Both arms left {full}x",
         _tile(_make_action(left_dx=-1, right_dx=-1))),

        # 2) Both arms go right
        ("02_both_right",
         f"Both arms right {full}x",
         _tile(_make_action(left_dx=+1, right_dx=+1))),

        # 3) Both arms go up
        ("03_both_up",
         f"Both arms up {full}x",
         _tile(_make_action(left_dy=+1, right_dy=+1))),

        # 4) Both arms go down
        ("04_both_down",
         f"Both arms down {full}x",
         _tile(_make_action(left_dy=-1, right_dy=-1))),

        # 5) Both arms left then both arms up (half-and-half)
        ("05_both_left_then_up",
         f"Both arms: left {half}x then up {half}x",
         _concat_halves(
             _make_action(left_dx=-1, right_dx=-1),
             _make_action(left_dy=+1, right_dy=+1),
         )),

        # 6) Left-left + right-right, then left-up + right-down (half-and-half)
        ("06_mirror_lr_then_ud",
         f"Mirror: L-left R-right {half}x, then L-up R-down {half}x",
         _concat_halves(
             _make_action(left_dx=-1, right_dx=+1),
             _make_action(left_dy=+1, right_dy=-1),
         )),

        # 7) Left arm up, right arm down
        ("07_left_up_right_down",
         f"Left arm up {full}x, right arm down {full}x",
         _tile(_make_action(left_dy=+1, right_dy=-1))),

        # 8) Identical to scenario 7 (duplicate per user specification)
        ("08_left_up_right_down_v2",
         f"Left arm up {full}x, right arm down {full}x (same as 07)",
         _tile(_make_action(left_dy=+1, right_dy=-1))),

        # 9) L-up R-down, then L-right R-left (half-and-half)
        ("09_split_ud_then_rl",
         f"L-up R-down {half}x, then L-right R-left {half}x",
         _concat_halves(
             _make_action(left_dy=+1, right_dy=-1),
             _make_action(left_dx=+1, right_dx=-1),
         )),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for OOD scenario generation.

    Parameters mirror ``inference_open_h.py`` for consistency, minus the
    single-dataset mode flags (this script is multi-dataset only).
    """
    parser = argparse.ArgumentParser(
        description="Generate OOD action trajectory scenario videos for Open-H",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model ---
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")

    # --- Dataset ---
    parser.add_argument("--test_episodes_json", type=str, required=True,
                        help="Path to test_episodes.json (from print_test_datasets_and_episodes.py)")
    parser.add_argument("--episodes_per_dataset", type=int, default=1,
                        help="Number of random episodes per dataset (default: 1)")
    parser.add_argument("--exclude_datasets", type=str, nargs="+", default=None,
                        help="Dataset names to skip (e.g., --exclude_datasets suturebot_2)")

    # --- Inference ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance", type=float, default=0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Context parallel size (GPUs)")

    # --- OOD ---
    parser.add_argument("--ood_magnitude", type=float, default=1.0,
                        help="Directional perturbation magnitude in std units (default: 1.0)")
    parser.add_argument("--stats_episodes", type=int, default=20,
                        help="Max episodes to sample for action statistics (default: 20)")
    parser.add_argument("--depth_only", action="store_true",
                        help="Only generate the two depth (z-axis) OOD scenarios, skip all others")

    # --- Output ---
    parser.add_argument("--save_root", type=str, default="results/open_h_ood",
                        help="Output directory")
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Action statistics
# ---------------------------------------------------------------------------

def compute_action_statistics(
    dataset,
    episode_map: dict[int, list[tuple[int, int]]],
    timestep_interval: int,
    max_episodes: int = 20,
    max_chunks_per_episode: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute per-dimension mean and std of transformed actions from sampled episodes.

    Samples actions from the loaded dataset (test split) to estimate the action
    distribution.  These statistics determine the magnitude of directional moves
    for OOD trajectory construction.  Since actions are mean-std normalised, the
    expected mean ≈ 0 and std ≈ 1, but empirical computation handles edge cases
    (partial episodes, non-standard normalisation).

    The raw stacked action array is also returned so callers can perform
    further analysis (e.g., clustering for frequent-action scenarios).

    Args:
        dataset: Loaded ``WrappedLeRobotSingleDataset`` (test split).
        episode_map: Episode-to-index mapping from :func:`build_episode_index_map`.
        timestep_interval: Raw-frame stride for this embodiment.
        max_episodes: Maximum number of episodes to sample.
        max_chunks_per_episode: Maximum chunks to collect per episode.
        seed: Random seed for episode sampling.

    Returns:
        Tuple of ``(mean_action, std_action, raw_dim, stacked_actions)`` where
        the first two arrays have shape ``(raw_dim,)``, ``raw_dim`` is the
        un-padded action dim, and ``stacked_actions`` has shape ``(N, raw_dim)``.
    """
    rng = random.Random(seed)

    usable_eps = [
        ep for ep in episode_map
        if any(bi == 0 for _, bi in episode_map[ep])
    ]

    n_sample = min(max_episodes, len(usable_eps))
    sampled = rng.sample(usable_eps, n_sample)

    all_actions: list[np.ndarray] = []
    raw_dim = 0

    for ep_id in sampled:
        chunks = find_chunk_indices(episode_map, ep_id, timestep_interval)
        if chunks is None:
            continue
        for ds_idx in chunks[:max_chunks_per_episode]:
            data = dataset[ds_idx]
            actions = data["action"]
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            all_actions.append(actions)
            if raw_dim == 0:
                raw_dim = actions.shape[-1]

    if not all_actions:
        raise RuntimeError("No actions collected — cannot compute statistics")

    stacked = np.concatenate(all_actions, axis=0)  # (N_steps, raw_dim)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std = np.maximum(std, 1e-6)  # guard against zero-variance dims

    logger.info(
        f"  Action statistics from {len(all_actions)} chunks ({len(stacked)} steps): "
        f"raw_dim={raw_dim}, mean_L2={np.linalg.norm(mean):.4f}, "
        f"std_range=[{std.min():.4f}, {std.max():.4f}]"
    )

    return mean, std, raw_dim, stacked


# ---------------------------------------------------------------------------
# Frequent-action discovery via k-means clustering
# ---------------------------------------------------------------------------

def _find_frequent_actions(
    stacked_actions: np.ndarray,
    n_frequent: int = 3,
    n_clusters: int = 10,
    max_iter: int = 30,
    seed: int = 42,
) -> list[tuple[np.ndarray, int]]:
    """Find the most frequently occurring action patterns via k-means clustering.

    Groups all sampled action steps into ``n_clusters`` clusters using Lloyd's
    algorithm, then returns the centroids of the ``n_frequent`` largest clusters
    (i.e., the action patterns that the most training steps are closest to).

    Args:
        stacked_actions: All sampled action steps, shape ``(N, D)``.
        n_frequent: Number of top clusters to return.
        n_clusters: Total number of k-means clusters.
        max_iter: Maximum Lloyd iterations.
        seed: Random seed for centroid initialisation.

    Returns:
        List of ``(centroid, cluster_size)`` tuples sorted by descending
        cluster size.  ``centroid`` has shape ``(D,)``.
    """
    rng = np.random.RandomState(seed)
    n = len(stacked_actions)

    if n <= n_clusters:
        n_clusters = max(n_frequent, n)

    indices = rng.choice(n, size=n_clusters, replace=False)
    centroids = stacked_actions[indices].copy()

    for _ in range(max_iter):
        # Compute distance from each point to each centroid (memory-efficient)
        dists = np.empty((n, n_clusters), dtype=np.float64)
        for k in range(n_clusters):
            dists[:, k] = np.linalg.norm(stacked_actions - centroids[k], axis=1)
        assignments = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_clusters, dtype=np.int64)
        for k in range(n_clusters):
            mask = assignments == k
            count = mask.sum()
            counts[k] = count
            if count > 0:
                new_centroids[k] = stacked_actions[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        if np.allclose(centroids, new_centroids, atol=1e-8):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final assignment to get accurate counts
    dists = np.empty((n, n_clusters), dtype=np.float64)
    for k in range(n_clusters):
        dists[:, k] = np.linalg.norm(stacked_actions - centroids[k], axis=1)
    assignments = np.argmin(dists, axis=1)
    counts = np.array([np.sum(assignments == k) for k in range(n_clusters)])

    top_k = np.argsort(-counts)[:n_frequent]
    results = [(centroids[k].copy(), int(counts[k])) for k in top_k]

    for rank, (centroid, count) in enumerate(results):
        logger.info(
            f"  Frequent action #{rank + 1}: cluster size={count}/{n} "
            f"({100 * count / n:.1f}%), L2={np.linalg.norm(centroid):.4f}"
        )

    return results


def _build_frequent_action_scenarios(
    frequent_actions: list[tuple[np.ndarray, int]],
    total_steps: int,
) -> list[tuple[str, str, np.ndarray]]:
    """Build OOD trajectories by repeating frequently occurring actions.

    Each centroid is tiled ``OOD_TRAJECTORY_LENGTH`` times (currently 60).

    Args:
        frequent_actions: List of ``(centroid, cluster_size)`` from
            :func:`_find_frequent_actions`.
        total_steps: Total number of action steps that were clustered
            (for percentage reporting in descriptions).

    Returns:
        List of ``(filename, description, trajectory)`` tuples where
        *trajectory* has shape ``(OOD_TRAJECTORY_LENGTH, D)``.
    """
    scenarios: list[tuple[str, str, np.ndarray]] = []

    for rank, (centroid, count) in enumerate(frequent_actions):
        pct = 100.0 * count / total_steps if total_steps > 0 else 0.0
        filename = f"{10 + rank:02d}_frequent_pattern_{rank + 1}"
        description = (
            f"Frequent action #{rank + 1} repeated {OOD_TRAJECTORY_LENGTH}x "
            f"(cluster {count}/{total_steps} steps, {pct:.1f}%, "
            f"L2={np.linalg.norm(centroid):.4f})"
        )
        trajectory = np.tile(centroid, (OOD_TRAJECTORY_LENGTH, 1))
        scenarios.append((filename, description, trajectory))

    return scenarios


def _build_depth_scenarios(
    arm_layout: dict,
    mean_action: np.ndarray,
    std_action: np.ndarray,
    magnitude: float = 1.0,
) -> list[tuple[str, str, np.ndarray]]:
    """Build two depth OOD trajectories: push into tissue (+z) and pull away (-z).

    Both arms are moved simultaneously along the z (depth) axis for the full
    trajectory length.  Positive z is defined as *into* the tissue/material,
    negative z as *away from* it.

    Args:
        arm_layout: Layout dict with ``left_z``, ``right_z``, ``dual``, etc.
        mean_action: Per-dimension mean of transformed actions, shape ``(D,)``.
        std_action: Per-dimension std, shape ``(D,)``.
        magnitude: Perturbation scale in standard-deviation units.

    Returns:
        List of two ``(filename, description, trajectory)`` tuples.
    """
    full = OOD_TRAJECTORY_LENGTH
    lz = arm_layout.get("left_z")
    rz = arm_layout.get("right_z")
    is_dual = arm_layout["dual"]

    def _make_depth_action(dz: float) -> np.ndarray:
        a = mean_action.copy()
        if lz is not None:
            a[lz] += dz * magnitude * std_action[lz]
        if is_dual and rz is not None:
            a[rz] += dz * magnitude * std_action[rz]
        return a

    return [
        ("13_depth_push_into",
         f"Both arms push into tissue (+z) {full}x",
         np.tile(_make_depth_action(+1.0), (full, 1))),

        ("14_depth_pull_away",
         f"Both arms pull away from tissue (-z) {full}x",
         np.tile(_make_depth_action(-1.0), (full, 1))),
    ]


# ---------------------------------------------------------------------------
# Trajectory rollout
# ---------------------------------------------------------------------------

def rollout_trajectory(
    video2world: ActionVideo2WorldInference,
    initial_frame: np.ndarray,
    padded_actions: np.ndarray,
    num_chunks: int,
    seed: int,
    guidance: float,
) -> np.ndarray:
    """Generate a video by rolling out actions through the model autoregressively.

    Each chunk feeds the model one conditioning frame and ``CHUNK_SIZE`` actions
    to produce ``CHUNK_SIZE`` predicted frames.  Chunks are stitched so the
    conditioning frame appears only once (at the start).

    Args:
        video2world: Loaded inference pipeline.
        initial_frame: Conditioning frame, shape ``(H, W, C)`` uint8.
        padded_actions: Pre-padded action trajectory, shape
            ``(num_chunks * CHUNK_SIZE, MAX_ACTION_DIM)`` float.
        num_chunks: Number of 12-frame inference chunks.
        seed: Base random seed (incremented per chunk).
        guidance: Classifier-free guidance scale.

    Returns:
        Stitched video as uint8 array. First chunk retains all frames
        (conditioning + 12 predicted), subsequent chunks contribute 12 frames.
        Total length: ``1 + num_chunks * CHUNK_SIZE``.
    """
    current_frame = initial_frame
    chunks: list[np.ndarray] = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        chunk_actions = padded_actions[start : start + CHUNK_SIZE]

        torch.cuda.synchronize()
        next_frame, video_chunk = video2world.step_inference(
            img_array=current_frame,
            action=chunk_actions.astype(np.float32),
            guidance=guidance,
            seed=seed + chunk_idx,
            num_latent_conditional_frames=1,
        )
        torch.cuda.synchronize()

        chunks.append(video_chunk)
        current_frame = next_frame

    stitched = [chunks[0]]
    for c in chunks[1:]:
        stitched.append(c[1:])
    return np.concatenate(stitched, axis=0)


def _save_short_reference_videos(
    gt_video: np.ndarray,
    predicted_video: np.ndarray,
    save_dir: str,
    episode_id: int,
    save_fps: int,
) -> None:
    """Save short GT/predicted clips matching OOD length for downstream composition.

    Clips contain the first ``SHORT_PRED_FRAMES`` predicted frames plus the
    initial conditioning frame (``SHORT_TOTAL_FRAMES`` total), so they align
    frame-for-frame with the OOD scenario videos.

    Args:
        gt_video: Full stitched GT video, shape ``(T, H, W, C)``.
        predicted_video: Full stitched predicted video, shape ``(T, H, W, C)``.
        save_dir: Dataset output directory.
        episode_id: Episode index.
        save_fps: Output FPS.
    """
    gt_short = gt_video[:SHORT_TOTAL_FRAMES]
    pred_short = predicted_video[:SHORT_TOTAL_FRAMES]

    if len(gt_short) < SHORT_TOTAL_FRAMES or len(pred_short) < SHORT_TOTAL_FRAMES:
        logger.warning(
            f"Episode {episode_id}: short reference videos have fewer than "
            f"{SHORT_TOTAL_FRAMES} frames "
            f"(gt={len(gt_short)}, predicted={len(pred_short)})."
        )

    subdir = f"gt_{SHORT_PRED_FRAMES}pred"
    gt_short_path = os.path.join(save_dir, subdir, f"episode_{episode_id:04d}.mp4")
    subdir = f"predicted_{SHORT_PRED_FRAMES}pred"
    pred_short_path = os.path.join(save_dir, subdir, f"episode_{episode_id:04d}.mp4")
    os.makedirs(os.path.dirname(gt_short_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_short_path), exist_ok=True)

    mediapy.write_video(gt_short_path, gt_short, fps=save_fps)
    mediapy.write_video(pred_short_path, pred_short, fps=save_fps)
    logger.info(
        f"  Saved {SHORT_PRED_FRAMES}-pred-frame references "
        f"(gt={len(gt_short)} frames, predicted={len(pred_short)} frames)"
    )


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

def run_episode_ood(
    video2world: ActionVideo2WorldInference,
    dataset,
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    timestep_interval: int,
    arm_layout: dict,
    mean_action: np.ndarray,
    std_action: np.ndarray,
    raw_dim: int,
    frequent_actions: list[tuple[np.ndarray, int]],
    total_action_steps: int,
    save_dir: str,
    seed: int,
    guidance: float,
    save_fps: int,
    ood_magnitude: float,
    depth_only: bool = False,
) -> int:
    """Process one episode: save GT video, predicted video, and all OOD scenarios.

    Produces three categories of output under *save_dir*:
      * ``gt/episode_XXXX.mp4`` — ground-truth video from the dataset.
      * ``predicted/episode_XXXX.mp4`` — model prediction with GT actions.
      * ``ood_scenarios/episode_XXXX/<scenario>.mp4`` — directional + frequent
        action OOD rollouts.

    Args:
        video2world: Loaded inference pipeline.
        dataset: Loaded dataset with transforms applied.
        episode_map: Episode-to-index mapping.
        episode_id: Episode to process.
        timestep_interval: Raw-frame stride for this embodiment.
        arm_layout: Position index mapping for each arm's x/y.
        mean_action: Per-dimension action mean, shape ``(raw_dim,)``.
        std_action: Per-dimension action std, shape ``(raw_dim,)``.
        raw_dim: Un-padded action dimensionality.
        frequent_actions: Top-k frequent action centroids with cluster sizes
            from :func:`_find_frequent_actions`.
        total_action_steps: Total action steps used for clustering (for
            percentage labels in metadata).
        save_dir: Output directory for this dataset.
        seed: Random seed for inference.
        guidance: Classifier-free guidance scale.
        save_fps: FPS for saved videos.
        ood_magnitude: Perturbation magnitude in std units.
        depth_only: If True, only generate the two depth (z-axis) OOD
            scenarios and skip directional + frequent-action scenarios.

    Returns:
        Number of OOD videos generated (0 if episode was skipped).
    """
    chunk_indices = find_chunk_indices(episode_map, episode_id, timestep_interval)
    if chunk_indices is None or not chunk_indices:
        logger.warning(f"Episode {episode_id}: no usable chunks, skipping")
        return 0

    gt_chunk_count = min(NUM_GT_CHUNKS, len(chunk_indices))
    if gt_chunk_count < NUM_GT_CHUNKS:
        logger.warning(
            f"Episode {episode_id}: only {gt_chunk_count} chunks available "
            f"(need {NUM_GT_CHUNKS} for full 72-frame GT)"
        )
    chunk_indices_gt = chunk_indices[:gt_chunk_count]
    logger.info(f"Episode {episode_id}: {gt_chunk_count} chunks for GT/predicted")

    # ------------------------------------------------------------------
    # 1. Collect GT frames and actions from the dataset
    # ------------------------------------------------------------------
    gt_frame_chunks: list[np.ndarray] = []
    gt_action_chunks: list[np.ndarray] = []
    initial_frame = None

    for chunk_idx, ds_idx in enumerate(chunk_indices_gt):
        data = dataset[ds_idx]

        video = data["video"]
        if isinstance(video, torch.Tensor):
            video = video.permute(1, 2, 3, 0).numpy()  # (C,T,H,W) → (T,H,W,C)
        elif video.ndim == 4 and video.shape[0] == 3:
            video = np.transpose(video, (1, 2, 3, 0))

        raw_actions = data["action"]
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.numpy()

        if chunk_idx == 0:
            initial_frame = video[0].copy()
            logger.info(
                f"  Video shape: {video.shape}, "
                f"action dim: {raw_actions.shape[-1]}D (raw) → {MAX_ACTION_DIM}D (padded)"
            )

        gt_frame_chunks.append(video)
        gt_action_chunks.append(raw_actions)

    # Stitch GT video: first chunk all frames, subsequent drop frame 0
    gt_stitched = [gt_frame_chunks[0]]
    for c in gt_frame_chunks[1:]:
        gt_stitched.append(c[1:])
    gt_video = np.concatenate(gt_stitched, axis=0)

    # ------------------------------------------------------------------
    # 2. Save GT video
    # ------------------------------------------------------------------
    gt_path = os.path.join(save_dir, "gt", f"episode_{episode_id:04d}.mp4")
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    mediapy.write_video(gt_path, gt_video, fps=save_fps)
    logger.info(f"  Saved GT video ({len(gt_video)} frames) → {gt_path}")

    # ------------------------------------------------------------------
    # 3. Predicted video with GT actions (autoregressive)
    # ------------------------------------------------------------------
    current_frame = initial_frame
    predicted_chunks: list[np.ndarray] = []

    for chunk_idx, raw_actions in enumerate(gt_action_chunks):
        padded = pad_action(raw_actions, MAX_ACTION_DIM)

        torch.cuda.synchronize()
        next_frame, video_chunk = video2world.step_inference(
            img_array=current_frame,
            action=padded.astype(np.float32),
            guidance=guidance,
            seed=seed + chunk_idx,
            num_latent_conditional_frames=1,
        )
        torch.cuda.synchronize()

        predicted_chunks.append(video_chunk)
        current_frame = next_frame

    pred_stitched = [predicted_chunks[0]]
    for c in predicted_chunks[1:]:
        pred_stitched.append(c[1:])
    predicted_video = np.concatenate(pred_stitched, axis=0)

    pred_path = os.path.join(save_dir, "predicted", f"episode_{episode_id:04d}.mp4")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    mediapy.write_video(pred_path, predicted_video, fps=save_fps)
    logger.info(f"  Saved predicted video ({len(predicted_video)} frames) → {pred_path}")

    # Save short reference clips matching OOD length (initial frame + first 60 predicted).
    _save_short_reference_videos(
        gt_video=gt_video,
        predicted_video=predicted_video,
        save_dir=save_dir,
        episode_id=episode_id,
        save_fps=save_fps,
    )

    # ------------------------------------------------------------------
    # 4. OOD scenario videos
    # ------------------------------------------------------------------
    depth_scenarios = _build_depth_scenarios(
        arm_layout, mean_action, std_action, ood_magnitude,
    )

    if depth_only:
        all_ood_scenarios = depth_scenarios
    else:
        directional_scenarios = _build_ood_scenarios(
            arm_layout, mean_action, std_action, ood_magnitude,
        )
        frequent_scenarios = _build_frequent_action_scenarios(
            frequent_actions, total_action_steps,
        )
        all_ood_scenarios = directional_scenarios + frequent_scenarios + depth_scenarios

    ood_dir = os.path.join(save_dir, "ood_scenarios", f"episode_{episode_id:04d}")
    os.makedirs(ood_dir, exist_ok=True)

    scenario_meta: list[dict] = []

    for scenario_idx, (filename, description, trajectory) in enumerate(all_ood_scenarios):
        padded_traj = pad_action(trajectory, MAX_ACTION_DIM)

        t0 = time.perf_counter()
        ood_video = rollout_trajectory(
            video2world=video2world,
            initial_frame=initial_frame,
            padded_actions=padded_traj,
            num_chunks=NUM_OOD_CHUNKS,
            seed=seed + scenario_idx * 1000,
            guidance=guidance,
        )
        elapsed = time.perf_counter() - t0

        ood_path = os.path.join(ood_dir, f"{filename}.mp4")
        mediapy.write_video(ood_path, ood_video, fps=save_fps)
        logger.info(
            f"  OOD {filename}: {len(ood_video)} frames, {elapsed:.1f}s — {description}"
        )

        scenario_meta.append({
            "filename": f"{filename}.mp4",
            "description": description,
            "frames": int(len(ood_video)),
            "seed": seed + scenario_idx * 1000,
            "inference_time_s": round(elapsed, 2),
        })

    # Save per-episode OOD metadata
    meta_path = os.path.join(ood_dir, "scenarios.json")
    with open(meta_path, "w") as f:
        json.dump({
            "episode_id": int(episode_id),
            "ood_magnitude": ood_magnitude,
            "arm_layout": arm_layout,
            "action_stats": {
                "raw_dim": int(raw_dim),
                "mean_l2": float(np.linalg.norm(mean_action)),
                "std_range": [float(std_action.min()), float(std_action.max())],
            },
            "scenarios": scenario_meta,
        }, f, indent=2)

    return len(all_ood_scenarios)


# ---------------------------------------------------------------------------
# Multi-dataset orchestration
# ---------------------------------------------------------------------------

def run_multi_dataset(args: argparse.Namespace, video2world: ActionVideo2WorldInference) -> None:
    """Run GT, predicted, and OOD generation across all datasets in test_episodes.json.

    For each dataset, randomly samples ``--episodes_per_dataset`` episodes
    (controlled by ``--seed``) and generates:
      * Ground-truth video (from dataset frames).
      * Predicted video with GT actions (autoregressive model inference).
      * Up to 14 OOD scenario videos per episode (9 directional + 3 frequent-action
        + 2 depth), or 2 depth-only with ``--depth_only``.

    Results are saved under ``save_root/<timestamp>/<dataset_name>/``.
    """
    with open(args.test_episodes_json, "r") as f:
        test_episodes: dict = json.load(f)

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_root = os.path.join(args.save_root, timestamp)
    os.makedirs(run_root, exist_ok=True)

    logger.info(f"OOD scenario generation — saving to {run_root}")
    logger.info(
        f"Seed: {args.seed} | Episodes/dataset: {args.episodes_per_dataset} | "
        f"OOD magnitude: {args.ood_magnitude} | Stats episodes: {args.stats_episodes}"
    )

    excluded = set(args.exclude_datasets) if args.exclude_datasets else set()
    if excluded:
        logger.info(f"Excluding datasets: {sorted(excluded)}")

    run_meta: dict = {
        "timestamp": timestamp,
        "seed": args.seed,
        "episodes_per_dataset": args.episodes_per_dataset,
        "ood_magnitude": args.ood_magnitude,
        "stats_episodes": args.stats_episodes,
        "ckpt_path": args.ckpt_path,
        "experiment": args.experiment,
        "test_episodes_json": args.test_episodes_json,
        "exclude_datasets": sorted(excluded),
        "datasets": {},
    }

    total_start = time.perf_counter()
    total_episodes = 0
    total_ood_videos = 0

    for dataset_name, info in test_episodes.items():
        if dataset_name in excluded:
            logger.info(f"[{dataset_name}] Skipped (excluded)")
            continue

        embodiment = info["embodiment"]
        path = info["path"]
        available_episodes = info["episode_ids"]

        if not available_episodes:
            logger.warning(f"[{dataset_name}] No test episodes available, skipping")
            continue

        exclude_splits = _lookup_exclude_splits(path)
        timestep_interval = resolve_timestep_interval(embodiment)

        logger.info("=" * 70)
        logger.info(f"[{dataset_name}] embodiment={embodiment}, path={path}")
        logger.info("=" * 70)

        try:
            dataset = load_wrapped_dataset(
                path=path, embodiment=embodiment,
                data_split="test", exclude_splits=exclude_splits,
            )
        except Exception as e:
            logger.error(f"[{dataset_name}] Failed to load dataset: {e}")
            raise

        episode_map = build_episode_index_map(dataset)
        timestep_stride = CHUNK_SIZE * timestep_interval

        # Only keep episodes that start at base_index=0 and have enough
        # chunks for full OOD rollout + matching GT/predicted references.
        min_chunks_needed = max(NUM_OOD_CHUNKS, 2)  # at least NUM_OOD_CHUNKS (5)
        max_base_index = (min_chunks_needed - 1) * timestep_stride
        usable_episodes = [
            ep for ep in available_episodes
            if ep in episode_map
            and any(bi == 0 for _, bi in episode_map[ep])
            and any(bi >= max_base_index for _, bi in episode_map[ep])
        ]

        logger.info(
            f"[{dataset_name}] timestep_interval={timestep_interval}, "
            f"{len(usable_episodes)} usable of {len(available_episodes)} available episodes "
            f"(require ≥{min_chunks_needed} chunks)"
        )

        if not usable_episodes:
            logger.error(f"[{dataset_name}] No usable episodes found, skipping")
            continue

        # --- Compute action statistics and find frequent actions ---
        mean_action, std_action, raw_dim, stacked_actions = compute_action_statistics(
            dataset, episode_map, timestep_interval,
            max_episodes=args.stats_episodes, seed=args.seed,
        )
        frequent_actions = _find_frequent_actions(
            stacked_actions, n_frequent=3, seed=args.seed,
        )
        total_action_steps = len(stacked_actions)

        # --- Resolve arm layout ---
        emb_str = embodiment.value if hasattr(embodiment, "value") else str(embodiment)
        arm_layout = ARM_LAYOUTS.get(emb_str)
        if arm_layout is None:
            arm_layout = _auto_detect_arm_layout(emb_str, raw_dim)

        logger.info(f"[{dataset_name}] arm_layout: {arm_layout}")

        n_pick = min(args.episodes_per_dataset, len(usable_episodes))
        selected = sorted(rng.sample(usable_episodes, n_pick))
        logger.info(f"[{dataset_name}] Selected episodes: {selected}")

        ds_save_dir = os.path.join(run_root, dataset_name)
        ds_meta = {
            "embodiment": emb_str,
            "path": path,
            "arm_layout": arm_layout,
            "raw_dim": raw_dim,
            "selected_episodes": selected,
        }

        for episode_id in selected:
            logger.info(f"[{dataset_name}] Processing episode {episode_id}")
            ep_start = time.perf_counter()

            try:
                n_ood = run_episode_ood(
                    video2world=video2world,
                    dataset=dataset,
                    episode_map=episode_map,
                    episode_id=episode_id,
                    timestep_interval=timestep_interval,
                    arm_layout=arm_layout,
                    mean_action=mean_action,
                    std_action=std_action,
                    raw_dim=raw_dim,
                    frequent_actions=frequent_actions,
                    total_action_steps=total_action_steps,
                    save_dir=ds_save_dir,
                    seed=args.seed,
                    guidance=args.guidance,
                    save_fps=args.save_fps,
                    ood_magnitude=args.ood_magnitude,
                    depth_only=args.depth_only,
                )
            except Exception as e:
                logger.error(f"[{dataset_name}] Error on episode {episode_id}: {e}")
                import traceback
                traceback.print_exc()
                raise

            ep_time = time.perf_counter() - ep_start
            if n_ood > 0:
                total_episodes += 1
                total_ood_videos += n_ood
                logger.info(
                    f"[{dataset_name}] Episode {episode_id} complete in {ep_time:.1f}s "
                    f"(GT + predicted + {n_ood} OOD scenarios)"
                )
            else:
                logger.warning(f"[{dataset_name}] Episode {episode_id} failed")

        run_meta["datasets"][dataset_name] = ds_meta

    # --- Save global run metadata ---
    meta_path = os.path.join(run_root, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info(f"Run metadata saved to {meta_path}")

    total_time = time.perf_counter() - total_start
    logger.info("=" * 70)
    logger.info(
        f"DONE — {total_episodes} episodes, {total_ood_videos} OOD videos, "
        f"{total_time:.1f}s total"
    )
    logger.info(f"Output: {run_root}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: parse args, load model, run OOD generation."""
    torch.set_grad_enabled(False)
    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Loading model from {args.ckpt_path}")
    video2world = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )
    mem_bytes = torch.cuda.memory_allocated()
    logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")

    run_multi_dataset(args, video2world)

    video2world.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()

