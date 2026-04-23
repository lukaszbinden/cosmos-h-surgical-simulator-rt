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

Usage
-----
Convert every subset listed in ``jhu_robot_base_dir_list_to_convert``::

    python scripts/convert_jhu_zarr_to_lerobot.py \
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono

Convert a single subset::

    python scripts/convert_jhu_zarr_to_lerobot.py \
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono \
        --subsets /knot_tying

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
import re
import shutil
import sys
import time
import traceback
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
from lerobot.datasets.utils import write_info


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


def _stack_row(struct_arr: np.ndarray, row_idx: int, names: Sequence[str]) -> np.ndarray:
    """Project a structured numpy record at ``row_idx`` onto the ordered fields in ``names``."""
    return np.array([struct_arr[n][row_idx] for n in names], dtype=np.float32)


def process_episode(
    dataset: LeRobotDataset,
    episode: EpisodeRef,
) -> int:
    """Stream a single zarr episode into ``dataset``; returns number of frames appended.

    The caller is responsible for calling ``dataset.save_episode()`` after this
    function returns (or ``dataset.clear_episode_buffer()`` on failure).
    """
    store = zarr.ZipStore(str(episode.zip_path), mode="r")
    try:
        zg = zarr.group(store=store)

        if "kinematics" not in zg:
            raise RuntimeError(
                f"zarr store missing 'kinematics' array: {episode.zip_path}"
            )
        if "left" not in zg:
            raise RuntimeError(
                f"zarr store missing 'left' array: {episode.zip_path}"
            )

        kinematics: np.ndarray = zg["kinematics"][:]
        missing_state = [n for n in STATES_NAME if n not in kinematics.dtype.names]
        missing_action = [n for n in ACTIONS_NAME if n not in kinematics.dtype.names]
        if missing_state or missing_action:
            raise RuntimeError(
                f"zarr store {episode.zip_path} kinematics is missing "
                f"fields: state={missing_state}, action={missing_action}"
            )

        left_arr = zg["left"]
        n_kin = len(kinematics)
        n_img = left_arr.shape[0]
        num_frames = min(n_kin, n_img)
        if num_frames == 0:
            raise RuntimeError(
                f"zarr store {episode.zip_path}: no frames "
                f"(kinematics={n_kin}, left={n_img})"
            )

        # Decode + resize all frames in one pass.  This is cheaper than
        # doing it inside the per-frame loop because zarr's JPEG decode
        # already paid the CPU cost per chunk.
        frames = np.asarray(left_arr[:num_frames])
        frames = maybe_resize_frames(frames)

        for i in range(num_frames):
            state_vec = _stack_row(kinematics, i, STATES_NAME)
            action_vec = _stack_row(kinematics, i, ACTIONS_NAME)

            # Use frame_index / fps so timestamps start at 0 and increase
            # strictly monotonically with sub-second precision — this is what
            # validate_formatting.validate_timestamps() expects.
            timestamp_sec = i / float(FPS)

            frame = {
                VIDEO_FEATURE_KEY: frames[i],
                "observation.state": state_vec,
                "action": action_vec,
                "instruction.text": episode.instruction,
                "observation.meta.tool.psm1": "Large Needle Driver",
                "observation.meta.tool.psm2": "Debakey Forceps",
            }
            dataset.add_frame(
                frame, task=episode.instruction, timestamp=timestamp_sec
            )

        return num_frames
    finally:
        store.close()


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


def maybe_write_splits(dataset: LeRobotDataset, num_episodes: int) -> None:
    """Add default 80/10/10 train/val/test splits to ``info.json``.

    Adding splits is a best-practice recommendation in the Open-H validator
    (WARNING-level otherwise).  For datasets with only one episode we skip
    this entirely since LeRobot validates the slices.
    """
    if num_episodes < 3:
        return
    train_count = max(1, int(0.8 * num_episodes))
    val_count = max(1, int(0.1 * num_episodes))
    # Put the remainder in test to avoid truncation.
    test_start = train_count + val_count
    dataset.meta.info["splits"] = {
        "train": f"0:{train_count}",
        "val": f"{train_count}:{test_start}",
        "test": f"{test_start}:{num_episodes}",
    }
    write_info(dataset.meta.info, dataset.root)


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
) -> Path:
    """Convert a set of episode zips into a single LeRobot dataset directory."""
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"[{dataset_name}] no episodes to convert")

    final_output_path = _resolve_output_path(output_root, dataset_name)
    print(f"[{dataset_name}] output: {final_output_path}")

    if overwrite and final_output_path.exists():
        print(f"[{dataset_name}] removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

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

    total_frames = 0
    saved_episodes = 0
    start_time = time.time()
    pbar = tqdm(episodes, desc=f"[{dataset_name}] episodes", unit="ep")
    for ep in pbar:
        try:
            n = process_episode(dataset, ep)
            dataset.save_episode()
            total_frames += n
            saved_episodes += 1
            pbar.set_postfix(frames=total_frames, saved=saved_episodes)
        except Exception as e:
            print(f"[{dataset_name}] ERROR processing {ep.zip_path}: {e}")
            traceback.print_exc()
            dataset.clear_episode_buffer()
            continue

    elapsed = time.time() - start_time
    print(
        f"[{dataset_name}] converted {saved_episodes}/{len(episodes)} episodes "
        f"({total_frames} frames) in {elapsed:.1f}s"
    )

    if saved_episodes == 0:
        print(f"[{dataset_name}] no episodes saved, skipping post-processing")
        return final_output_path

    maybe_write_splits(dataset, saved_episodes)
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
        image_writer_processes: Parallel processes for video encoding.
        image_writer_threads: Threads per video-encoding process.
        batch_encoding_size: How many episodes to batch for MP4 encoding.
            ``1`` is safest but slower; larger values improve throughput at
            the cost of leaving trailing episodes as loose images if the
            total episode count isn't a multiple of the batch size.
        overwrite: If True, remove any existing dataset directory before
            writing.
    """
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
        )


if __name__ == "__main__":
    tyro.cli(main)
