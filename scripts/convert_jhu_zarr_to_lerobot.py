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
Convert JHU dVRK zarr-in-zip subsets to LeRobot v2.1 format.

This script iterates over the JHU surgical-robot subsets listed in
``jhu_robot_base_dir_list_to_convert`` (see
``exp605-nigeln-14_cosmos_predict2/cosmos_predict2/configs/action_conditioned/defaults/data.py``)
and converts each subset into its own LeRobot v2.1 dataset.  The resulting
datasets satisfy the format requirements enforced by
``open-h-embodiment/scripts/validation/validate_formatting.py`` and follow the
conventions expected by the ``jhu_dvrk_mono`` embodiment in
Cosmos-H-Surgical-Simulator's Open-H training pipeline.

Input layout (per subset)
-------------------------
::

    <base_dir>/
        tissue_<N>/
            <idx>_<subtask_name>/
                <episode_id>.zip       # zarr store (see below)
                ...

Each ``.zip`` is a zarr store containing:

- ``kinematics`` — structured array with ``timestamp`` + dVRK pose / setpoint
  fields (``psm1_pose.position.x``, ..., ``psm1_jaw``, ``psm2_sp.orientation.w``,
  ``psm2_jaw_sp``, plus extra channels the model does not use).
- ``left`` — ``(N, H, W, 3)`` ``uint8`` array of JPEG-encoded left-endoscope
  frames (the codec id is ``pi_jpeg``; this script registers a compatible codec
  at import time).
- Optional ``right``, ``endo_psm1``, ``endo_psm2`` camera channels (ignored by
  default; ``jhu_dvrk_mono`` uses only the left endoscope).

Output layout (per subset)
--------------------------
::

    <output_root>/<dataset_name>/
        data/chunk-000/episode_*.parquet
        videos/chunk-000/observation.images.endoscope_left/episode_*.mp4
        meta/{info,episodes,episodes_stats,tasks}.jsonl
        meta/modality.json
        meta/stats.json
        meta/README.md

The resulting dataset satisfies the ``jhu_dvrk_mono`` expectations:

- state / action vectors are **raw 16D** (``psm1_pose (7) + psm1_gripper (1) +
  psm2_pose (7) + psm2_gripper (1)``); the ``GenericRelativeActionTransform``
  in ``groot_configs.py`` converts them to 20D at training time.
- videos are re-encoded at **960x540** (the default ``jhu_dvrk_mono`` cadence
  uses a 512x288 crop at training time).
- ``meta/modality.json`` maps ``state.psm1_pose``/``state.psm1_gripper``/
  ``state.psm2_pose``/``state.psm2_gripper`` (and the matching ``action.*``
  keys) to the correct index ranges, and exposes the left endoscope as
  ``video.endoscope_left``.

Parallelism
-----------
With ``--num-workers N`` (default: ``min(8, cpu_count()-1)``) the script
splits a subset's episodes into ``N`` round-robin shards and runs
``N`` independent LeRobot conversions in parallel — each shard has its own
ffmpeg/SVT-AV1 video encoder, which is where the real wall-clock time goes.
After all shards finish, the main process merges them into a single
LeRobot-v2.1 dataset by renaming parquet+MP4 files and concatenating the
JSONL metadata (with ``episode_index`` / ``task_index`` / ``index`` columns
rewritten to the merged numbering).

For ``--num-workers 1`` the script falls back to a purely sequential
in-process conversion (no merging).  Use this for debugging and on
memory-constrained machines.

Peak disk usage during shard-and-merge is roughly ``2x`` the final dataset
size (sharded copies + final output) until the merge finishes and the
shard directory is deleted.

Usage
-----
Convert every subset listed in ``jhu_robot_base_dir_list_to_convert``
(parallel by default)::

    python scripts/convert_jhu_zarr_to_lerobot.py \
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono \
        --num-workers 16

Convert a single subset sequentially::

    python scripts/convert_jhu_zarr_to_lerobot.py \
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono \
        --subsets /knot_tying \
        --num-workers 1

Debug on the bundled sample zip (flat-zip mode, one episode → one dataset)::

    python scripts/convert_jhu_zarr_to_lerobot.py \
        --single-zip data/20250117-132221-722360.zip \
        --output-root /tmp/lerobot_debug \
        --dataset-name sample_debug \
        --task "knot tying"

Dependencies
------------
- lerobot == 0.3.3
- zarr (v2 API; tested with 2.18.x)
- numcodecs + simplejpeg (for the ``pi_jpeg`` codec)
- numpy, pandas, opencv-python, pillow, tqdm, tyro
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numcodecs
import numpy as np
import pandas as pd
import simplejpeg
import tyro
import zarr
from numcodecs.abc import Codec
from tqdm import tqdm

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw state columns in the kinematics structured array (16D total).
# Order MUST match the indices referenced in ``meta/modality.json`` below.
STATES_NAME: list[str] = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
]

# Raw action columns in the kinematics structured array (16D total).
ACTIONS_NAME: list[str] = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]

# Target video resolution (width x height).
TARGET_WIDTH: int = 960
TARGET_HEIGHT: int = 540

# FPS of the JHU captures (see ``JhuTabletopDataset.fps_orig`` in
# ``exp605-nigeln-14_cosmos_predict2``).
FPS: int = 30

ROBOT_TYPE: str = "jhu_dvrk_mono"

# LeRobot feature key for the single monocular camera used by jhu_dvrk_mono.
VIDEO_FEATURE_KEY: str = "observation.images.endoscope_left"

# Short alias exposed to the Cosmos pipeline via ``meta/modality.json``:
# ``video.endoscope_left`` with original_key set to VIDEO_FEATURE_KEY.
VIDEO_MODALITY_KEY: str = "endoscope_left"

# Default draco/Cosmos prefix where the JHU zarr subsets live.
JHU_PREFIX_DEFAULT: str = (
    "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/JHU_data_jpeg100_noacc_clean++"
)

# Subsets to convert, copied from ``jhu_robot_base_dir_list_to_convert`` in
# ``exp605-nigeln-14_cosmos_predict2/.../defaults/data.py``.  Keep in sync.
SUBSETS_TO_CONVERT: list[str] = [
    "/knot_tying",
    "/suture_bot/success/processed_data_zipped_pi",
    "/suture_bot/failure/processed_data_zipped_pi",
    "/ood/processed_data_zipped_pi",
    "/cosmos_fail_filtered",
    "/cosmos_throw_fail_demo",
    "/cosmos_knot_fail_demo",
    "/suturebot_act_throw_eval",
]


# ---------------------------------------------------------------------------
# pi_jpeg codec registration
# ---------------------------------------------------------------------------


class _PiJpegCodec(Codec):
    """Matches the ``JpegCodec`` registered by
    ``cosmos_predict2.data.action_conditioned.jhu_data.image_utils``.

    We reproduce it here so the conversion script can run standalone without
    needing that repository on ``PYTHONPATH``.
    """

    codec_id = "pi_jpeg"

    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality

    def encode(self, buf: np.ndarray) -> bytes:
        assert buf.dtype == np.uint8, f"expected uint8, got {buf.dtype}"
        assert buf.ndim == 4 and buf.shape[0] == 1 and buf.shape[-1] == 3, (
            f"expected (1, H, W, 3), got {buf.shape}"
        )
        return simplejpeg.encode_jpeg(buf[0], quality=self.quality)

    def decode(self, buf, out=None):
        img = simplejpeg.decode_jpeg(buf, buffer=out)
        return img[np.newaxis, ...]


@functools.cache
def _register_pi_jpeg_codec() -> None:
    numcodecs.register_codec(_PiJpegCodec)


_register_pi_jpeg_codec()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def resize_with_padding(
    images: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Letterbox-resize ``(B, H, W, 3)`` images to ``(B, target_height, target_width, 3)``.

    This mirrors ``cosmos_predict2.data.action_conditioned.jhu_data.image_utils.resize_with_padding``
    so the converted videos match the aspect-ratio handling used by the
    ``JhuTabletopDataset`` dataloader at train time.
    """
    batch_size = images.shape[0]
    padded = np.zeros((batch_size, target_height, target_width, 3), dtype=np.uint8)
    target_aspect = target_width / target_height
    for i in range(batch_size):
        h, w = images[i].shape[:2]
        aspect = w / h
        if aspect > target_aspect:
            new_w = target_width
            new_h = max(1, int(round(new_w / aspect)))
        else:
            new_h = target_height
            new_w = max(1, int(round(new_h * aspect)))
        resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2
        padded[i, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded


def maybe_resize_frames(frames: np.ndarray) -> np.ndarray:
    """Resize ``frames`` (B, H, W, 3) to ``(B, 540, 960, 3)`` if needed."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected (B, H, W, 3) uint8 frames, got shape={frames.shape}"
        )
    if frames.shape[1] == TARGET_HEIGHT and frames.shape[2] == TARGET_WIDTH:
        return frames.astype(np.uint8, copy=False)
    return resize_with_padding(frames, TARGET_WIDTH, TARGET_HEIGHT)


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRef:
    """A discovered episode, ready to convert."""

    zip_path: Path
    instruction: str
    tissue_index: Optional[int]
    subtask_name: str  # the instruction-folder name (e.g. "1_knot_tying")
    is_recovery: bool


def _clean_instruction_name(folder_name: str) -> tuple[str, bool]:
    """Reproduce the instruction-name cleanup used by
    ``cosmos_predict2.data.action_conditioned.jhu_data.episode_utils.get_robot_episodes``.

    Returns ``(instruction, is_recovery)``.
    """
    instr = re.sub(r"^\d+_*", "", folder_name)
    instr = re.sub(r"\d+", "", instr)
    is_recovery = "recovery" in instr
    instr = instr.replace("recovery", "")
    instr = instr.replace("_", " ")
    instr = instr.strip()
    return instr, is_recovery


def discover_episodes(base_dir: Path) -> list[EpisodeRef]:
    """Discover all episode zip files under a JHU subset root.

    Expected layout::

        <base_dir>/tissue_<N>/<idx>_<subtask>/<episode>.zip
    """
    episodes: list[EpisodeRef] = []
    if not base_dir.exists():
        print(f"[WARN] base directory does not exist: {base_dir}")
        return episodes

    tissue_dirs = sorted(
        p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("tissue_")
    )
    for tissue_dir in tissue_dirs:
        m = re.search(r"tissue_(\d+)", tissue_dir.name)
        tissue_index = int(m.group(1)) if m else None
        for instr_dir in sorted(p for p in tissue_dir.iterdir() if p.is_dir()):
            instruction, is_recovery = _clean_instruction_name(instr_dir.name)
            for zp in sorted(
                p for p in instr_dir.iterdir() if p.is_file() and p.suffix == ".zip"
            ):
                episodes.append(
                    EpisodeRef(
                        zip_path=zp,
                        instruction=instruction,
                        tissue_index=tissue_index,
                        subtask_name=instr_dir.name,
                        is_recovery=is_recovery,
                    )
                )

    episodes.sort(key=lambda ep: str(ep.zip_path))
    return episodes


def resolve_dataset_name(subset_rel_path: str) -> str:
    """Derive an output dataset name from a relative subset path.

    Examples
    --------
    >>> resolve_dataset_name("/knot_tying")
    'knot_tying'
    >>> resolve_dataset_name("/suture_bot/success/processed_data_zipped_pi")
    'suture_bot_success'
    >>> resolve_dataset_name("/ood/processed_data_zipped_pi")
    'ood'
    """
    parts = [p for p in subset_rel_path.strip("/").split("/") if p]
    # Drop purely descriptive trailing suffixes.
    drop_parts = {
        "processed_data_zipped_pi",
        "processed_suturing_data_zipped_pi",
        "processed_suturing_data_zipped_pi_clean",
    }
    cleaned = [p for p in parts if p not in drop_parts]
    if not cleaned:
        cleaned = parts
    return "_".join(cleaned)


# ---------------------------------------------------------------------------
# Zarr → LeRobot episode
# ---------------------------------------------------------------------------


@dataclass
class EpisodePayload:
    """Decoded episode data produced by a worker process.

    Workers return one of these per episode.  All large-array fields use
    ``numpy.ndarray`` so that multiprocessing's default pickle-based IPC path
    can serialize them efficiently.  For a typical 520-frame 960x540 episode
    the ``frames`` array weighs ~800MB, so keep ``num_workers`` modest on
    memory-constrained machines.
    """

    episode: EpisodeRef
    frames: np.ndarray  # (num_frames, TARGET_HEIGHT, TARGET_WIDTH, 3) uint8
    states: np.ndarray  # (num_frames, len(STATES_NAME)) float32
    actions: np.ndarray  # (num_frames, len(ACTIONS_NAME)) float32
    error: Optional[str] = None

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


def _empty_payload(episode: EpisodeRef, error: str) -> EpisodePayload:
    """Helper: construct an empty payload tagged with a worker-side error."""
    return EpisodePayload(
        episode=episode,
        frames=np.zeros((0, TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8),
        states=np.zeros((0, len(STATES_NAME)), dtype=np.float32),
        actions=np.zeros((0, len(ACTIONS_NAME)), dtype=np.float32),
        error=error,
    )


def _stack_columns(struct_arr: np.ndarray, num_rows: int, names: Sequence[str]) -> np.ndarray:
    """Vectorize ``names`` columns of a structured array into ``(num_rows, len(names))`` float32."""
    return np.column_stack(
        [struct_arr[n][:num_rows].astype(np.float32, copy=False) for n in names]
    )


def _decode_episode(episode: EpisodeRef) -> EpisodePayload:
    """Worker-side: open a zarr-in-zip, decode frames + kinematics, return payload.

    Catches *all* exceptions and returns them on ``payload.error`` rather than
    raising.  This keeps the consumer loop (which calls ``dataset.save_episode``
    in the main process) simple: it just checks ``payload.error`` before
    ingesting and logs + skips on failure.

    Safe to call in a ``multiprocessing.Pool`` worker.  Module-level
    ``_register_pi_jpeg_codec()`` runs at import time so the ``pi_jpeg`` codec
    is available in every worker regardless of the start method (fork/spawn).
    """
    try:
        store = zarr.ZipStore(str(episode.zip_path), mode="r")
        try:
            zg = zarr.group(store=store)

            missing = [k for k in ("kinematics", "left") if k not in zg]
            if missing:
                raise RuntimeError(f"zarr store missing arrays: {missing}")

            kinematics: np.ndarray = zg["kinematics"][:]
            missing_state = [n for n in STATES_NAME if n not in kinematics.dtype.names]
            missing_action = [n for n in ACTIONS_NAME if n not in kinematics.dtype.names]
            if missing_state or missing_action:
                raise RuntimeError(
                    f"kinematics missing fields: "
                    f"state={missing_state}, action={missing_action}"
                )

            left_arr = zg["left"]
            n_kin = len(kinematics)
            n_img = left_arr.shape[0]
            num_frames = min(n_kin, n_img)
            if num_frames == 0:
                raise RuntimeError(f"no frames (kinematics={n_kin}, left={n_img})")

            frames = np.asarray(left_arr[:num_frames])
            frames = maybe_resize_frames(frames)

            states = _stack_columns(kinematics, num_frames, STATES_NAME)
            actions = _stack_columns(kinematics, num_frames, ACTIONS_NAME)

            return EpisodePayload(
                episode=episode,
                frames=frames,
                states=states,
                actions=actions,
                error=None,
            )
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001 — worker is the last line of defence
        return _empty_payload(episode, f"{type(e).__name__}: {e}")


def _ingest_payload(dataset: LeRobotDataset, payload: EpisodePayload) -> int:
    """Main-process: push a decoded payload into the LeRobotDataset buffer.

    Raises ``RuntimeError`` if ``payload.error`` is set, so the caller can
    branch into ``dataset.clear_episode_buffer()``.  The caller is responsible
    for calling ``dataset.save_episode()`` on success.
    """
    if payload.error:
        raise RuntimeError(
            f"worker failed on {payload.episode.zip_path}: {payload.error}"
        )
    num_frames = payload.num_frames
    if num_frames == 0:
        raise RuntimeError(f"payload is empty: {payload.episode.zip_path}")

    ep = payload.episode
    psm1_tool = "Large Needle Driver"
    psm2_tool = "Debakey Forceps"
    fps_f = float(FPS)
    for i in range(num_frames):
        # Use frame_index / fps so timestamps start at 0 and increase
        # strictly monotonically with sub-second precision — this is what
        # validate_formatting.validate_timestamps() expects.
        frame = {
            VIDEO_FEATURE_KEY: payload.frames[i],
            "observation.state": payload.states[i],
            "action": payload.actions[i],
            "instruction.text": ep.instruction,
            "observation.meta.tool.psm1": psm1_tool,
            "observation.meta.tool.psm2": psm2_tool,
        }
        dataset.add_frame(frame, task=ep.instruction, timestamp=i / fps_f)
    return num_frames


def process_episode(
    dataset: LeRobotDataset,
    episode: EpisodeRef,
) -> int:
    """Sequential convenience wrapper: decode then ingest one episode.

    Identical in effect to calling ``_decode_episode`` followed by
    ``_ingest_payload``, but exposed as a single function for backward
    compatibility and for the ``num_workers <= 1`` code path.  The caller is
    responsible for ``dataset.save_episode()`` / ``clear_episode_buffer()``.
    """
    payload = _decode_episode(episode)
    return _ingest_payload(dataset, payload)


# ---------------------------------------------------------------------------
# modality.json / README.md / stats.json
# ---------------------------------------------------------------------------


def _state_action_modality_entries(original_key: str) -> dict[str, dict]:
    """Return the per-subkey modality entries shared by state and action.

    The offsets match the order in ``STATES_NAME`` / ``ACTIONS_NAME``.
    Together they cover indices [0, 16) of the flat 16D state/action vector.
    """
    return {
        "psm1_pose": {
            "start": 0,
            "end": 7,
            "rotation_type": "quaternion",
            "absolute": True,
            "dtype": "float32",
            "range": None,
            "original_key": original_key,
        },
        "psm1_gripper": {
            "start": 7,
            "end": 8,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
            "original_key": original_key,
        },
        "psm2_pose": {
            "start": 8,
            "end": 15,
            "rotation_type": "quaternion",
            "absolute": True,
            "dtype": "float32",
            "range": None,
            "original_key": original_key,
        },
        "psm2_gripper": {
            "start": 15,
            "end": 16,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
            "original_key": original_key,
        },
    }


def build_modality_json() -> dict:
    """Build the ``meta/modality.json`` payload for a ``jhu_dvrk_mono`` dataset.

    Shape matches the ``LeRobotModalityMetadata`` schema in
    ``cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.schema``.
    """
    return {
        "state": _state_action_modality_entries("observation.state"),
        "action": _state_action_modality_entries("action"),
        "video": {
            VIDEO_MODALITY_KEY: {
                "original_key": VIDEO_FEATURE_KEY,
            },
        },
        "annotation": {
            "instruction.text": {"original_key": "instruction.text"},
        },
    }


def write_modality_json(dataset_path: Path) -> None:
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "modality.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_modality_json(), f, indent=2)
        f.write("\n")
    print(f"[OK] wrote {path}")


def write_readme(
    dataset_path: Path,
    dataset_name: str,
    source_path: str,
    num_episodes: int,
    num_frames: int,
) -> None:
    """Write a minimal ``meta/README.md`` satisfying validate_formatting.py's checks.

    The validator looks for ``synchron``/``timestamp`` in the README to confirm
    that synchronization was considered — the text below mentions both.
    """
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    readme = meta_dir / "README.md"
    content = f"""# {dataset_name}

LeRobot v2.1 dataset converted from JHU dVRK zarr subset at **{source_path}**.

- Embodiment: `{ROBOT_TYPE}` (dVRK JHU monocular, left endoscope only)
- FPS: `{FPS}` (native capture rate)
- Video resolution: `{TARGET_WIDTH}x{TARGET_HEIGHT}` (re-encoded via letterbox
  aspect-preserving resize to match `JhuTabletopDataset`'s in-pipeline behavior)
- Episodes: `{num_episodes}`
- Frames: `{num_frames}`

## Modalities

- `observation.images.endoscope_left` — left stereo endoscope, re-encoded MP4.
- `observation.state` — 16D float32: `[psm1_pose (xyz+quat_xyzw), psm1_jaw,`
  `psm2_pose (xyz+quat_xyzw), psm2_jaw]`.
- `action` — 16D float32, same layout as state but using the teleop setpoint
  fields (`psm1_sp.*`, `psm1_jaw_sp`, `psm2_sp.*`, `psm2_jaw_sp`).
- `instruction.text` — natural-language subtask description derived from the
  source subtask folder name.
- `observation.meta.tool.psm1` / `observation.meta.tool.psm2` — constant tool
  metadata (`Large Needle Driver` / `Debakey Forceps`) for the JHU dVRK rig.

The per-subkey slicing (`state.psm1_pose` etc.) is described in
`meta/modality.json` and is the single source of truth consumed by the Cosmos
`LeRobotSingleDataset`.

## Synchronization

Timestamps are generated as `frame_index / fps` (seconds, relative to episode
start) so per-frame deltas are preserved at full float64 precision.  The raw
kinematics stream already arrives time-aligned with the left-endoscope frames
in the source zarr (the two arrays share the same leading dimension); we
truncate both to `min(len(kinematics), len(left))` before emitting frames.

## Action space note

The 7D per-arm pose uses the raw `xyz + quat_xyzw` layout.  Cosmos-H training
converts this to `xyz_rel + rot6d_rel` (9D per arm) via
`GenericRelativeActionTransform` — **do not** pre-convert to 6D rotations when
using this dataset with the Cosmos `jhu_dvrk_mono` pipeline.
"""
    with readme.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] wrote {readme}")


def _compute_vec_stats(vec: np.ndarray) -> dict:
    return {
        "mean": vec.mean(axis=0).astype(np.float64).tolist(),
        "std": vec.std(axis=0).astype(np.float64).tolist(),
        "min": vec.min(axis=0).astype(np.float64).tolist(),
        "max": vec.max(axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(vec, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(vec, 0.99, axis=0).astype(np.float64).tolist(),
    }


def write_stats_json(dataset_path: Path) -> None:
    """Aggregate per-episode parquets into a dataset-level ``meta/stats.json``.

    This is the *raw* column-wise statistics.  Note that training uses
    ``meta/stats_cosmos.json`` (computed by
    ``scripts/compute_openh_action_stats.py``) for post-transform stats;
    ``stats.json`` is still useful for sanity checks and for downstream tools
    that expect LeRobot-standard statistics.
    """
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        print(f"[WARN] no parquet files found in {data_dir}, skipping stats.json")
        return

    all_state: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "observation.state" in df.columns:
            all_state.append(np.vstack(df["observation.state"].to_numpy()))
        if "action" in df.columns:
            all_action.append(np.vstack(df["action"].to_numpy()))

    stats: dict[str, dict] = {}
    if all_state:
        stats["observation.state"] = _compute_vec_stats(np.vstack(all_state))
    if all_action:
        stats["action"] = _compute_vec_stats(np.vstack(all_action))

    out = dataset_path / "meta" / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print(f"[OK] wrote {out}")


def maybe_write_splits_to_path(dataset_path: Path, num_episodes: int) -> None:
    """Add default 80/10/10 train/val/test splits to ``info.json``.

    Operates on the on-disk ``info.json`` directly so it can run after both
    the in-process sequential path and the shard-and-merge path (where we no
    longer hold a ``LeRobotDataset`` instance).  For datasets with <3
    episodes we skip this entirely.
    """
    if num_episodes < 3:
        return
    train_count = max(1, int(0.8 * num_episodes))
    val_count = max(1, int(0.1 * num_episodes))
    # Put the remainder in test to avoid truncation.
    test_start = train_count + val_count
    info_path = dataset_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["splits"] = {
        "train": f"0:{train_count}",
        "val": f"{train_count}:{test_start}",
        "test": f"{test_start}:{num_episodes}",
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)


# ---------------------------------------------------------------------------
# Per-subset conversion
# ---------------------------------------------------------------------------


def _features_spec() -> dict:
    """Features dict passed to ``LeRobotDataset.create``."""
    return {
        VIDEO_FEATURE_KEY: {
            "dtype": "video",
            "shape": (TARGET_HEIGHT, TARGET_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATES_NAME),),
            "names": STATES_NAME,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTIONS_NAME),),
            "names": ACTIONS_NAME,
        },
        "observation.meta.tool.psm1": {
            "dtype": "string",
            "shape": (1,),
            "names": ["value"],
        },
        "observation.meta.tool.psm2": {
            "dtype": "string",
            "shape": (1,),
            "names": ["value"],
        },
        "instruction.text": {
            "dtype": "string",
            "shape": (1,),
            "description": "Natural-language command for the robot",
        },
    }


def _resolve_output_path(output_root: Optional[Path], dataset_name: str) -> Path:
    """Decide where the dataset should be written.

    If ``output_root`` is given, the final path is ``output_root / dataset_name``.
    Otherwise fall back to ``$HF_LEROBOT_HOME/<dataset_name>``.  The caller
    passes the returned path to ``LeRobotDataset.create(root=...)``.
    """
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root / dataset_name
    return Path(HF_LEROBOT_HOME) / dataset_name


def default_num_workers(cap: int = 8) -> int:
    """Pick a sensible default for ``--num-workers``.

    Reserves one core for the main process (which does the merge) and caps at
    ``cap`` to avoid over-subscription on big boxes.  Users can override via
    ``--num-workers``.
    """
    cpu = os.cpu_count() or 1
    return max(1, min(cap, cpu - 1))


# ---------------------------------------------------------------------------
# Sequential conversion (used both directly for num_workers=1 and inside each
# worker process in shard-and-merge mode).
# ---------------------------------------------------------------------------


def _sequential_convert(
    episodes: list[tuple[int, EpisodeRef]],
    *,
    dataset: LeRobotDataset,
    log_prefix: str,
    show_progress: bool = True,
) -> list[dict]:
    """Run a sequential decode+ingest+save loop over ``episodes``.

    ``episodes`` is a list of ``(orig_idx, episode)`` pairs so that each saved
    episode can be traced back to its position in the parent subset's
    discovery order — that field drives the global ordering in
    :func:`_merge_shards_into`.

    Returns a list of per-episode metadata dicts for the episodes that were
    successfully saved (in save order, so local episode_index in the
    underlying ``dataset`` equals the list position).
    """
    results: list[dict] = []
    pbar_iter = (
        tqdm(episodes, desc=log_prefix, unit="ep")
        if show_progress
        else episodes
    )
    for orig_idx, episode in pbar_iter:
        payload: Optional[EpisodePayload] = None
        try:
            payload = _decode_episode(episode)
            n = _ingest_payload(dataset, payload)
            dataset.save_episode()
            results.append(
                {
                    "orig_idx": int(orig_idx),
                    "local_idx": len(results),
                    "num_frames": int(n),
                    "instruction": episode.instruction,
                    "zip_path": str(episode.zip_path),
                }
            )
        except Exception as e:  # noqa: BLE001 — last line of defence per episode
            ep_path = (
                payload.episode.zip_path if payload is not None else episode.zip_path
            )
            print(f"{log_prefix} ERROR on {ep_path}: {e}")
            traceback.print_exc()
            dataset.clear_episode_buffer()
            continue
    return results


# ---------------------------------------------------------------------------
# Shard-and-merge parallel conversion.
#
# Why shard-and-merge instead of a producer/consumer decode pool?
# LeRobot's ``save_episode`` is synchronous and dominated by the ffmpeg/
# SVT-AV1 video encode of the episode's frames.  Running K workers that each
# own their own ``LeRobotDataset`` lets K ffmpeg encodes progress in
# parallel, which is the actual wall-clock bottleneck.  The main process then
# does a pure-metadata merge (rename parquet+mp4 files, rewrite a few
# columns, concatenate JSONL files) which is cheap compared to any encode.
# ---------------------------------------------------------------------------


def _round_robin_shards(
    items: Sequence[tuple[int, EpisodeRef]], num_shards: int
) -> list[list[tuple[int, EpisodeRef]]]:
    """Split ``items`` into ``num_shards`` roughly-equal-size groups.

    Round-robin (index ``% num_shards``) keeps average episode length roughly
    balanced across shards even when the source order has runs of long or
    short episodes.
    """
    shards: list[list[tuple[int, EpisodeRef]]] = [
        [] for _ in range(num_shards)
    ]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return [s for s in shards if s]


def _worker_convert_shard(
    shard_idx: int,
    shard_items: list[tuple[int, EpisodeRef]],
    shard_root: str,
    shard_dataset_name: str,
    image_writer_processes: int,
    image_writer_threads: int,
    tolerance_s: float,
    batch_encoding_size: int,
) -> dict:
    """Worker process: convert one shard to a standalone LeRobot dataset.

    Each worker creates its own ``LeRobotDataset`` at
    ``<shard_root>/dataset/`` so that its ffmpeg video encoder runs
    independently from every other shard's.  The returned dict is pickled
    back to the main process, which uses it to rename/rewrite files in
    :func:`_merge_shards_into`.

    ``shard_root`` is passed as a ``str`` (not ``Path``) so that pickling is
    trivially stable across Python versions.
    """
    shard_root_path = Path(shard_root)
    shard_root_path.mkdir(parents=True, exist_ok=True)
    dataset_path = shard_root_path / "dataset"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    dataset = LeRobotDataset.create(
        repo_id=shard_dataset_name,
        root=dataset_path,
        use_videos=True,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=_features_spec(),
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        tolerance_s=tolerance_s,
        batch_encoding_size=batch_encoding_size,
    )

    log_prefix = f"[shard_{shard_idx:04d}]"
    start = time.time()
    # Don't print tqdm bars from inside workers — they'd clobber each other
    # (and the main progress bar) in the terminal.
    results = _sequential_convert(
        shard_items,
        dataset=dataset,
        log_prefix=log_prefix,
        show_progress=False,
    )
    elapsed = time.time() - start

    print(
        f"{log_prefix} saved {len(results)}/{len(shard_items)} episodes "
        f"in {elapsed:.1f}s"
    )
    return {
        "shard_idx": int(shard_idx),
        "shard_root": str(shard_root_path),
        "results": results,
        "elapsed": float(elapsed),
    }


def _chunks_size_from_shard(shard_root: Path) -> int:
    """Read ``chunks_size`` from a shard's ``info.json`` (default 1000)."""
    info_path = shard_root / "dataset" / "meta" / "info.json"
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return int(info.get("chunks_size", 1000))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 1000


def _update_constant_stat(stat: dict, value: float) -> None:
    """In-place update a LeRobot per-episode stat that's a single scalar.

    For constant-per-episode columns (``episode_index``, ``task_index``),
    LeRobot writes ``min == max == mean == value`` and ``std == 0``.  After
    remapping the value we just need to refresh those fields.  LeRobot stores
    each as either a scalar or a 1-element list depending on version — handle
    both shapes.
    """
    for k in ("min", "max", "mean"):
        if k in stat:
            stat[k] = [value] if isinstance(stat[k], list) else value
    if "std" in stat:
        stat["std"] = [0.0] if isinstance(stat["std"], list) else 0.0


def _update_index_stat(stat: dict, first: int, last: int) -> None:
    """Refresh the stats for the ``index`` column (global frame index).

    After merging shards the global frame range for this episode moves, so
    ``min``/``max``/``mean``/``std`` need to be recomputed from the new
    ``arange(first, last+1)``.
    """
    n = last - first + 1
    if n <= 0:
        return
    mean = (first + last) / 2.0
    std = float(np.std(np.arange(n))) if n > 1 else 0.0
    for k, v in (("min", first), ("max", last), ("mean", mean)):
        if k in stat:
            stat[k] = [v] if isinstance(stat[k], list) else v
    if "std" in stat:
        stat["std"] = [std] if isinstance(stat["std"], list) else std


def _merge_shards_into(
    shard_metas: list[dict],
    final_output: Path,
    *,
    chunks_size: int = 1000,
    dataset_name: str = "",
) -> tuple[int, int, list[str]]:
    """Merge sharded LeRobot datasets into a single dataset at ``final_output``.

    Steps (all done by the main process, no workers):

    1. Flatten per-episode records across shards and sort by ``orig_idx`` so
       the merged dataset's episode order matches the original discovery
       order exactly (i.e. ``episode_000000.parquet`` is the first episode
       discovered by :func:`discover_episodes`).
    2. Build a global ``tasks`` list in first-seen order and a
       ``task_to_idx`` lookup.
    3. Pre-load every shard's ``episodes_stats.jsonl`` into an in-memory
       lookup keyed by ``(shard_idx, local_idx)``.
    4. For each merged episode:
         - rewrite its parquet into the final ``data/chunk-*/`` tree with
           the new ``episode_index``, ``index`` (global frame id offset),
           and ``task_index`` columns;
         - ``shutil.move`` its MP4 into the final ``videos/chunk-*/...``
           tree under the new filename;
         - remap the corresponding ``episodes_stats.jsonl`` line to the new
           indices.
    5. Write merged ``episodes.jsonl``, ``episodes_stats.jsonl``,
       ``tasks.jsonl``, and ``info.json`` (using the first shard's
       ``info.json`` as a template for everything except the counts and
       splits, which we recompute).

    Returns ``(total_episodes, total_frames, tasks_in_order)``.
    """
    # 1. Flatten + sort.
    flat: list[tuple[int, dict]] = []
    for sm in shard_metas:
        for r in sm["results"]:
            flat.append((sm["shard_idx"], r))
    flat.sort(key=lambda sr: sr[1]["orig_idx"])

    if not flat:
        raise RuntimeError(
            f"[{dataset_name}] all shards were empty — nothing to merge"
        )

    # 2. Global tasks list.
    tasks_in_order: list[str] = []
    task_to_idx: dict[str, int] = {}
    for _, entry in flat:
        t = entry["instruction"]
        if t not in task_to_idx:
            task_to_idx[t] = len(tasks_in_order)
            tasks_in_order.append(t)

    # 3. Preload per-shard episodes_stats.
    shard_roots: dict[int, Path] = {
        sm["shard_idx"]: Path(sm["shard_root"]) for sm in shard_metas
    }
    episodes_stats_lookup: dict[tuple[int, int], dict] = {}
    for sm in shard_metas:
        stats_path = shard_roots[sm["shard_idx"]] / "dataset/meta/episodes_stats.jsonl"
        if not stats_path.exists():
            continue
        with stats_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episodes_stats_lookup[(sm["shard_idx"], int(rec["episode_index"]))] = rec

    # Pre-create final dirs.
    final_output.mkdir(parents=True, exist_ok=True)
    (final_output / "data").mkdir(exist_ok=True)
    (final_output / "videos").mkdir(exist_ok=True)
    (final_output / "meta").mkdir(exist_ok=True)

    episodes_jsonl: list[dict] = []
    episodes_stats_out: list[dict] = []
    total_frames = 0
    cumulative_idx = 0

    pbar = tqdm(flat, desc=f"[{dataset_name}] merge", unit="ep")
    for new_idx, (shard_idx, entry) in enumerate(pbar):
        local_idx = int(entry["local_idx"])
        shard_ds = shard_roots[shard_idx] / "dataset"

        old_chunk = local_idx // chunks_size
        src_parquet = (
            shard_ds / f"data/chunk-{old_chunk:03d}/episode_{local_idx:06d}.parquet"
        )
        src_video = (
            shard_ds
            / f"videos/chunk-{old_chunk:03d}/{VIDEO_FEATURE_KEY}/"
            f"episode_{local_idx:06d}.mp4"
        )

        new_chunk = new_idx // chunks_size
        dst_parquet = (
            final_output / f"data/chunk-{new_chunk:03d}/episode_{new_idx:06d}.parquet"
        )
        dst_video = (
            final_output
            / f"videos/chunk-{new_chunk:03d}/{VIDEO_FEATURE_KEY}/"
            f"episode_{new_idx:06d}.mp4"
        )
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        dst_video.parent.mkdir(parents=True, exist_ok=True)

        if not src_parquet.exists():
            raise RuntimeError(
                f"[{dataset_name}] merge: missing source parquet {src_parquet} "
                f"(shard {shard_idx}, local_idx {local_idx})"
            )
        if not src_video.exists():
            raise RuntimeError(
                f"[{dataset_name}] merge: missing source video {src_video}"
            )

        # Rewrite parquet with new indices.
        df = pd.read_parquet(src_parquet)
        n = len(df)
        df["episode_index"] = np.int64(new_idx)
        df["index"] = np.arange(
            cumulative_idx, cumulative_idx + n, dtype=df["index"].dtype
        )
        new_task_idx = task_to_idx[entry["instruction"]]
        df["task_index"] = np.int64(new_task_idx)
        df.to_parquet(dst_parquet, engine="pyarrow")
        # We've re-written the data, so the shard-local file is no longer needed.
        src_parquet.unlink(missing_ok=True)

        # Videos are pure visual data — pure rename is enough.
        shutil.move(str(src_video), str(dst_video))

        # episodes.jsonl entry.
        episodes_jsonl.append(
            {
                "episode_index": int(new_idx),
                "tasks": [entry["instruction"]],
                "length": int(n),
            }
        )

        # episodes_stats.jsonl entry (remap indices).
        stats_rec = episodes_stats_lookup.get((shard_idx, local_idx))
        if stats_rec is not None:
            stats_rec = json.loads(json.dumps(stats_rec))  # deep-copy
            stats_rec["episode_index"] = int(new_idx)
            stats_map = stats_rec.get("stats", {})
            if isinstance(stats_map.get("task_index"), dict):
                _update_constant_stat(stats_map["task_index"], float(new_task_idx))
            if isinstance(stats_map.get("episode_index"), dict):
                _update_constant_stat(stats_map["episode_index"], float(new_idx))
            if isinstance(stats_map.get("index"), dict):
                _update_index_stat(
                    stats_map["index"], cumulative_idx, cumulative_idx + n - 1
                )
            episodes_stats_out.append(stats_rec)

        total_frames += n
        cumulative_idx += n

        pbar.set_postfix(frames=total_frames, saved=new_idx + 1)
    pbar.close()

    # Write JSONL files.
    with (final_output / "meta/episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in episodes_jsonl:
            f.write(json.dumps(rec) + "\n")

    with (final_output / "meta/episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for rec in episodes_stats_out:
            f.write(json.dumps(rec) + "\n")

    with (final_output / "meta/tasks.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks_in_order):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    # Base info.json from any shard, then overwrite the counts / splits.
    template_info_path = shard_roots[shard_metas[0]["shard_idx"]] / "dataset/meta/info.json"
    with template_info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["total_episodes"] = len(flat)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks_in_order)
    info["total_videos"] = len(flat)
    info["total_chunks"] = max(1, (len(flat) + chunks_size - 1) // chunks_size)
    info["chunks_size"] = chunks_size
    # maybe_write_splits_to_path will update "splits" below; set a safe
    # single-split default in case we have <3 episodes.
    info["splits"] = {"train": f"0:{len(flat)}"}
    with (final_output / "meta/info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    return len(flat), total_frames, tasks_in_order


def convert_subset(
    episodes: list[EpisodeRef],
    dataset_name: str,
    output_root: Optional[Path],
    *,
    overwrite: bool = True,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    tolerance_s: float = 1.0 / FPS,
    batch_encoding_size: int = 1,
    max_episodes: Optional[int] = None,
    source_path_str: str = "",
    num_workers: int = 1,
) -> Path:
    """Convert a set of episode zips into a single LeRobot dataset directory.

    Parallelism
    -----------
    * ``num_workers <= 1``: a single in-process ``LeRobotDataset`` consumes
      episodes sequentially.  Useful for debugging / low-memory machines.
    * ``num_workers > 1``: **shard-and-merge**.  Episodes are split round-robin
      into ``num_workers`` shards; each shard is converted end-to-end in its
      own worker process (its own zarr decode, its own ffmpeg video
      encoder).  The main process then merges the shards into a single
      LeRobot dataset by renaming the parquet + MP4 files and concatenating
      the JSONL metadata.  This is the mode to use for large runs (e.g. the
      ~500-zip JHU subsets) since it actually parallelizes LeRobot's AV1
      video encode, which is the dominant wall-clock cost in the sequential
      path.

    Peak disk usage
    ---------------
    Shard-and-merge temporarily stores each shard's data+videos under a
    sibling directory of the final output (``<final_output>.shards``); peak
    disk usage is therefore roughly ``2x`` the final dataset size during the
    merge phase.  The sibling directory is removed after a successful merge.
    """
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"[{dataset_name}] no episodes to convert")

    final_output_path = _resolve_output_path(output_root, dataset_name)
    print(f"[{dataset_name}] output: {final_output_path}")

    if overwrite and final_output_path.exists():
        print(f"[{dataset_name}] removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    effective_workers = min(max(1, num_workers), len(episodes))

    saved_episodes = 0
    total_frames = 0
    start_time = time.time()

    if effective_workers <= 1:
        # Sequential: single LeRobotDataset in-process.  Still uses the same
        # _decode_episode + _ingest_payload helpers as the parallel path so
        # the behaviour is bit-for-bit identical (modulo ordering).
        print(f"[{dataset_name}] sequential mode (num_workers=1)")
        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            root=final_output_path,
            use_videos=True,
            robot_type=ROBOT_TYPE,
            fps=FPS,
            features=_features_spec(),
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            tolerance_s=tolerance_s,
            batch_encoding_size=batch_encoding_size,
        )
        indexed = list(enumerate(episodes))
        results = _sequential_convert(
            indexed,
            dataset=dataset,
            log_prefix=f"[{dataset_name}]",
            show_progress=True,
        )
        saved_episodes = len(results)
        total_frames = sum(int(r["num_frames"]) for r in results)
        chunks_size = int(dataset.meta.info.get("chunks_size", 1000))
    else:
        # Shard-and-merge.
        shards_root = final_output_path.parent / f"{final_output_path.name}.shards"
        if shards_root.exists():
            print(f"[{dataset_name}] cleaning stale shards dir {shards_root}")
            shutil.rmtree(shards_root)
        shards_root.mkdir(parents=True, exist_ok=True)

        indexed = list(enumerate(episodes))
        shards = _round_robin_shards(indexed, effective_workers)
        print(
            f"[{dataset_name}] shard-and-merge mode "
            f"(num_workers={effective_workers}, shards={len(shards)}, "
            f"episodes/shard≈{len(indexed) // len(shards)})"
        )

        # One LeRobotDataset per shard → one ffmpeg encoder per shard, all
        # running in parallel.  ``spawn`` avoids fd/lock inheritance issues
        # between LeRobot's own image_writer processes across workers.
        mp_ctx = multiprocessing.get_context("spawn")
        shard_metas: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=len(shards), mp_context=mp_ctx
        ) as pool:
            futs: list[Future[dict]] = []
            for s_idx, shard_items in enumerate(shards):
                shard_root = shards_root / f"shard_{s_idx:04d}"
                futs.append(
                    pool.submit(
                        _worker_convert_shard,
                        s_idx,
                        shard_items,
                        str(shard_root),
                        dataset_name,
                        image_writer_processes,
                        image_writer_threads,
                        tolerance_s,
                        batch_encoding_size,
                    )
                )
            pbar = tqdm(
                total=len(futs),
                desc=f"[{dataset_name}] shards",
                unit="shard",
            )
            for fut in as_completed(futs):
                try:
                    sm = fut.result()
                    shard_metas.append(sm)
                except Exception as e:  # noqa: BLE001
                    print(f"[{dataset_name}] ERROR from shard worker: {e}")
                    traceback.print_exc()
                pbar.update(1)
            pbar.close()

        shard_metas.sort(key=lambda sm: sm["shard_idx"])
        # Merge.
        chunks_size = _chunks_size_from_shard(
            Path(shard_metas[0]["shard_root"])
        ) if shard_metas else 1000
        n_saved, total_frames, _tasks = _merge_shards_into(
            shard_metas,
            final_output_path,
            chunks_size=chunks_size,
            dataset_name=dataset_name,
        )
        saved_episodes = n_saved

        # Cleanup shard temp.
        shutil.rmtree(shards_root, ignore_errors=True)

    elapsed = time.time() - start_time
    throughput_ep = saved_episodes / elapsed if elapsed > 0 else 0.0
    throughput_fps = total_frames / elapsed if elapsed > 0 else 0.0
    print(
        f"[{dataset_name}] converted {saved_episodes}/{len(episodes)} episodes "
        f"({total_frames} frames) in {elapsed:.1f}s "
        f"({throughput_ep:.2f} ep/s, {throughput_fps:.1f} fps)"
    )

    if saved_episodes == 0:
        print(f"[{dataset_name}] no episodes saved, skipping post-processing")
        return final_output_path

    maybe_write_splits_to_path(final_output_path, saved_episodes)
    write_modality_json(final_output_path)
    write_readme(
        final_output_path,
        dataset_name=dataset_name,
        source_path=source_path_str or str(final_output_path),
        num_episodes=saved_episodes,
        num_frames=total_frames,
    )
    write_stats_json(final_output_path)

    return final_output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_single_zip(
    *,
    single_zip: Path,
    output_root: Optional[Path],
    dataset_name: Optional[str],
    task: str,
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
    batch_encoding_size: int,
    max_episodes: Optional[int],
    num_workers: int,
) -> None:
    if not single_zip.exists():
        print(f"ERROR: --single-zip does not exist: {single_zip}")
        sys.exit(1)
    resolved_name = dataset_name or single_zip.stem
    episode = EpisodeRef(
        zip_path=single_zip,
        instruction=task,
        tissue_index=None,
        subtask_name="single_zip",
        is_recovery=False,
    )
    convert_subset(
        episodes=[episode],
        dataset_name=resolved_name,
        output_root=output_root,
        overwrite=overwrite,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
        max_episodes=max_episodes,
        source_path_str=str(single_zip),
        num_workers=num_workers,
    )


def _run_subsets(
    *,
    input_base: Path,
    output_root: Optional[Path],
    subsets: list[str],
    dataset_name: Optional[str],
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
    batch_encoding_size: int,
    max_episodes: Optional[int],
    num_workers: int,
) -> None:
    if not input_base.exists():
        print(f"ERROR: --input-base does not exist: {input_base}")
        sys.exit(1)

    results: list[tuple[str, Path, int]] = []  # (dataset_name, output_path, num_episodes)
    only_one = len(subsets) == 1 and dataset_name is not None
    for subset_rel in subsets:
        base_dir = (input_base / subset_rel.lstrip("/")).resolve()
        resolved_name = (
            dataset_name if only_one else resolve_dataset_name(subset_rel)
        )
        episodes = discover_episodes(base_dir)
        if not episodes:
            print(f"[{resolved_name}] no episodes discovered under {base_dir}")
            continue
        output_path = convert_subset(
            episodes=episodes,
            dataset_name=resolved_name,
            output_root=output_root,
            overwrite=overwrite,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
            max_episodes=max_episodes,
            source_path_str=str(base_dir),
            num_workers=num_workers,
        )
        results.append((resolved_name, output_path, len(episodes)))

    print("\n=== conversion summary ===")
    if not results:
        print("No subsets were converted.")
        return
    for name, path, n in results:
        print(f"  {name}: {n} episodes → {path}")


def main(
    input_base: Path = Path(JHU_PREFIX_DEFAULT),
    output_root: Optional[Path] = None,
    subsets: list[str] = list(SUBSETS_TO_CONVERT),
    dataset_name: Optional[str] = None,
    single_zip: Optional[Path] = None,
    task: str = "dvrk suturing task",
    max_episodes: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    batch_encoding_size: int = 1,
    overwrite: bool = True,
) -> None:
    """Convert JHU dVRK zarr subsets to LeRobot v2.1 format.

    Args:
        input_base: Root of the JHU zarr subsets (matches ``prefix`` in
            ``defaults/data.py``). Subsets are resolved relative to this root.
        output_root: Where to write the converted LeRobot datasets. Defaults
            to ``$HF_LEROBOT_HOME``. One subdirectory is created per subset.
        subsets: Relative subset paths (e.g. ``/knot_tying``). Default: every
            entry in ``jhu_robot_base_dir_list_to_convert``.
        dataset_name: Override the output dataset name for the FIRST subset
            (or the ``--single-zip`` mode). Ignored when converting multiple
            subsets.
        single_zip: Debug mode — convert one zarr-in-zip episode directly
            into a dataset, bypassing the tissue/subtask discovery. Set
            ``--task`` to choose the instruction string.
        task: Instruction text used in ``--single-zip`` mode.
        max_episodes: Limit the number of episodes converted per subset
            (debug).
        num_workers: Number of parallel shards for shard-and-merge
            conversion.  Each subset's episodes are split round-robin into
            ``num_workers`` shards, each shard is converted to a standalone
            LeRobotDataset in its own subprocess (with its own ffmpeg video
            encoder — the true wall-clock bottleneck), and the shards are
            merged into a single dataset at the end.  Defaults to
            ``min(8, os.cpu_count() - 1)``.  Pass ``1`` for a purely
            sequential in-process conversion (no multiprocessing, no merge).
            Peak disk usage during the merge step is roughly ``2x`` the
            final dataset size.
        image_writer_processes: Parallel processes for video encoding.
        image_writer_threads: Threads per video-encoding process.
        batch_encoding_size: How many episodes to batch for MP4 encoding.
            ``1`` is safest but slower; larger values improve throughput at
            the cost of leaving trailing episodes as loose images if the
            total episode count isn't a multiple of the batch size.
        overwrite: If True, remove any existing dataset directory before
            writing.
    """
    if num_workers is None:
        num_workers = default_num_workers()
    if num_workers < 1:
        num_workers = 1

    if single_zip is not None:
        _run_single_zip(
            single_zip=single_zip,
            output_root=output_root,
            dataset_name=dataset_name,
            task=task,
            overwrite=overwrite,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
            max_episodes=max_episodes,
            num_workers=num_workers,
        )
    else:
        _run_subsets(
            input_base=input_base,
            output_root=output_root,
            subsets=subsets,
            dataset_name=dataset_name,
            overwrite=overwrite,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
            max_episodes=max_episodes,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    tyro.cli(main)
