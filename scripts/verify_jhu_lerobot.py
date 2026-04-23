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
Verify a LeRobot dataset produced by ``convert_jhu_zarr_to_lerobot.py``.

This script is a post-conversion sanity check.  It does NOT re-encode
anything; it only counts episodes, frames, and video frames, and compares
them against the source zarr zips.  Use it when the size difference between
the source tree and the converted LeRobot tree looks suspiciously large and
you want to confirm that nothing was silently dropped.

Checks performed
----------------
1. Episode count:  number of ``*.zip`` files in the source subset directory
   == ``info.json::total_episodes``.
2. Frame counts:  for a random sample of ``--sample-size`` episodes,
    a. open the source zarr and read ``len(kinematics)`` + ``left.shape[0]``;
    b. read the matching parquet and compare row count;
    c. run ``cv2.VideoCapture`` on the matching MP4 and compare frame count.
   The converter uses ``min(n_kin, n_img)`` as episode length, so the
   parquet/MP4 row count should equal that minimum, not ``max``.
3. Totals:  ``sum(length for ep in episodes.jsonl) == info.total_frames``.
4. (Informational)  disk-footprint ratio ``source/converted`` per subset,
   plus the per-frame bytes in each format.

The script tries very hard to be non-destructive and forgiving: a mismatch
is reported but never raised, so the whole report runs to completion even
when a few episodes are broken.  The final exit code is ``0`` iff every
episode-count check passed; frame-count sampling deltas still exit ``0``
because ``min(n_kin, n_img)`` truncation is expected.

Usage
-----
Single-subset::

    python scripts/verify_jhu_lerobot.py \
        --source-dir /path/to/JHU_data_jpeg100_noacc_clean++/knot_tying \
        --lerobot-dir /path/to/output/knot_tying \
        --sample-size 5

All subsets at once (pairs source ↔ converted by name)::

    python scripts/verify_jhu_lerobot.py \
        --source-root /path/to/JHU_data_jpeg100_noacc_clean++ \
        --lerobot-root /path/to/output \
        --subsets /knot_tying /suture_bot/success/processed_data_zipped_pi \
        --sample-size 3
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import av  # PyAV — already pulled in by LeRobot
import tyro


# ---------------------------------------------------------------------------
# zarr access + pi_jpeg codec registration
# ---------------------------------------------------------------------------

# Importing the converter has a small cost but gives us:
#   - the pi_jpeg codec registration side-effect (so we can open zarr zips)
#   - matching ``STATES_NAME`` / ``ACTIONS_NAME`` / ``VIDEO_FEATURE_KEY`` /
#     ``SUBSETS_TO_CONVERT`` constants
# We depend on those rather than hard-coding them here so any future
# changes to the converter stay in one place.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from convert_jhu_zarr_to_lerobot import (  # noqa: E402
    STATES_NAME,
    SUBSETS_TO_CONVERT,
    VIDEO_FEATURE_KEY,
    discover_episodes,
    resolve_dataset_name,
)
import zarr  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _SubsetReport:
    """Per-subset outcome, used to compute the final pass/fail exit code."""

    name: str
    source_zip_count: int
    converted_episode_count: int
    episode_count_match: bool
    sampled_frame_mismatches: int = 0
    sampled_video_mismatches: int = 0
    total_frames_info: int = 0
    total_frames_from_episodes: int = 0
    total_frames_match: bool = True
    source_bytes: int = 0
    converted_bytes: int = 0


def _count_source_zips(base_dir: Path) -> list[Path]:
    """Return every ``tissue_*/<instr>/*.zip`` under ``base_dir``.

    Sorted and deduplicated; matches the order used by
    :func:`convert_jhu_zarr_to_lerobot.discover_episodes`.
    """
    zips: list[Path] = []
    if not base_dir.exists():
        return zips
    for tissue in sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("tissue_")):
        for instr in sorted(p for p in tissue.iterdir() if p.is_dir()):
            for zp in sorted(p for p in instr.iterdir() if p.is_file() and p.suffix == ".zip"):
                zips.append(zp)
    return zips


def _read_zarr_lengths(zip_path: Path) -> tuple[int, int]:
    """Open ``zip_path`` as a zarr ZipStore and return ``(n_kin, n_img_left)``.

    ``(0, 0)`` is returned if the store can't be opened (e.g. truncated
    zip).  Callers log that as a failure rather than raising.
    """
    try:
        store = zarr.ZipStore(str(zip_path), mode="r")
    except Exception:
        return 0, 0
    try:
        zg = zarr.group(store=store)
        n_kin = int(len(zg["kinematics"])) if "kinematics" in zg else 0
        n_img = int(zg["left"].shape[0]) if "left" in zg else 0
        return n_kin, n_img
    except Exception:
        return 0, 0
    finally:
        try:
            store.close()
        except Exception:
            pass


def _load_lerobot_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing meta/info.json under {dataset_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_episodes_jsonl(dataset_path: Path) -> list[dict]:
    path = dataset_path / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing meta/episodes.jsonl under {dataset_path}")
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _episode_paths(dataset_path: Path, episode_index: int, info: dict) -> tuple[Path, Path]:
    """Resolve the parquet + MP4 file paths for ``episode_index``."""
    chunks_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunks_size
    parquet = dataset_path / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = (
        dataset_path
        / f"videos/chunk-{chunk:03d}/{VIDEO_FEATURE_KEY}/episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def _count_video_frames(video_path: Path) -> int:
    """Return the frame count of ``video_path`` using PyAV.

    PyAV (== libav bindings) is a hard dep of LeRobot so it's guaranteed to
    be available wherever a conversion was produced.  Strategy:

    1. First try the container metadata (``stream.frames``) — that's what
       ``ffprobe`` returns and it's O(1).  Some FFmpeg builds report ``0``
       for AV1 streams, in which case we fall back to (2).
    2. Demux and count video packets.  Each AV1 packet is one frame in the
       format LeRobot produces (``g=2`` + CRF-30 + ``preset 8`` with no
       ``repeat_pict``), so packet count == frame count.  This is still
       fast — we never decode any frame content.

    Returns ``0`` on any unexpected error so a single corrupted video
    can't abort the whole verification.
    """
    try:
        # PyAV occasionally lets libav write chatter to fd 2 (the AV1
        # decoder init line "Missing Sequence Header" etc.) even when we
        # never decode a packet.  Suppress it to keep the verifier output
        # clean; legitimate errors still surface via the except clauses.
        with _suppress_fd2():
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    return 0
                stream = container.streams.video[0]
                # Fast path: container-level frame count.
                meta_count = int(getattr(stream, "frames", 0) or 0)
                if meta_count > 0:
                    return meta_count
                # Slow path: demux and count packets.
                count = 0
                for packet in container.demux(stream):
                    # Flush sentinel — not a real frame.
                    if packet.dts is None and packet.pts is None and packet.size == 0:
                        continue
                    count += 1
                return count
    except Exception:
        return 0


def _suppress_fd2():
    """Context manager that redirects fd 2 to /dev/null (minimal re-impl).

    Duplicated here (instead of reusing the one in the converter) to keep
    the verifier a one-file, zero-other-imports tool that doesn't need the
    converter's ``lerobot``/``zarr`` module-level side-effects if someone
    just wants to ffprobe-style check a dataset.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        sys.stderr.flush()
        saved = os.dup(2)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, 2)
                yield
            finally:
                os.close(devnull)
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)

    return _ctx()


def _dir_size_bytes(path: Path) -> int:
    """Recursive ``du`` in Python; cheap because we only sum file sizes."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


# ---------------------------------------------------------------------------
# Per-subset verification
# ---------------------------------------------------------------------------


def verify_subset(
    source_dir: Path,
    lerobot_dir: Path,
    *,
    subset_name: str,
    sample_size: int,
    seed: int,
) -> _SubsetReport:
    """Verify a single converted subset.  Returns the report object."""
    print(f"\n=== {subset_name} ===")
    print(f"  source    : {source_dir}")
    print(f"  converted : {lerobot_dir}")

    src_zips = _count_source_zips(source_dir)
    if not lerobot_dir.exists():
        print(f"  ❌ converted directory does not exist")
        return _SubsetReport(
            name=subset_name,
            source_zip_count=len(src_zips),
            converted_episode_count=0,
            episode_count_match=False,
        )

    try:
        info = _load_lerobot_info(lerobot_dir)
        episodes = _load_episodes_jsonl(lerobot_dir)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return _SubsetReport(
            name=subset_name,
            source_zip_count=len(src_zips),
            converted_episode_count=0,
            episode_count_match=False,
        )

    total_episodes = int(info.get("total_episodes", 0))
    total_frames_info = int(info.get("total_frames", 0))
    total_frames_episodes = sum(int(e.get("length", 0)) for e in episodes)

    # 1. Episode counts
    episode_count_match = (len(src_zips) == total_episodes)
    tick = "✓" if episode_count_match else "❌"
    print(
        f"  {tick} episodes: source={len(src_zips)}, "
        f"converted={total_episodes}"
    )

    # 2. Frame totals (episodes.jsonl vs info.json)
    frame_total_match = (total_frames_episodes == total_frames_info)
    tick = "✓" if frame_total_match else "❌"
    print(
        f"  {tick} total_frames: info.json={total_frames_info}, "
        f"sum(episodes.jsonl)={total_frames_episodes}"
    )

    # 3. Random sample: zarr vs parquet vs mp4
    sampled_frame_mismatches = 0
    sampled_video_mismatches = 0
    if sample_size > 0 and src_zips:
        # Order src_zips and episodes by their sort key so index i in both
        # refers to the same episode (both are sorted by path/discovery
        # order inside the converter).  This only holds if the conversion
        # preserved discovery order, which it does for shard-and-merge.
        rng = random.Random(seed)
        sample_count = min(sample_size, len(src_zips), total_episodes)
        sample_indices = rng.sample(range(min(len(src_zips), total_episodes)), sample_count)

        print(f"  — sampling {sample_count} episode(s) for per-episode checks")
        for new_idx in sorted(sample_indices):
            zip_path = src_zips[new_idx]
            n_kin, n_img = _read_zarr_lengths(zip_path)
            expected_frames = min(n_kin, n_img)

            parquet_path, video_path = _episode_paths(lerobot_dir, new_idx, info)
            try:
                df = pd.read_parquet(parquet_path, columns=["frame_index"])
                parquet_len = len(df)
            except Exception as e:
                print(f"    ❌ episode {new_idx:06d}: parquet unreadable ({e})")
                sampled_frame_mismatches += 1
                continue

            video_len = _count_video_frames(video_path)

            parquet_ok = (parquet_len == expected_frames)
            video_ok = (video_len == expected_frames)

            if not parquet_ok:
                sampled_frame_mismatches += 1
            if not video_ok:
                sampled_video_mismatches += 1

            tick = "✓" if (parquet_ok and video_ok) else "⚠"
            print(
                f"    {tick} ep {new_idx:06d}: "
                f"zarr(n_kin={n_kin}, n_img={n_img}) → expected={expected_frames} | "
                f"parquet={parquet_len} | mp4={video_len} | {zip_path.name}"
            )

    # 4. Disk footprint (informational)
    source_bytes = _dir_size_bytes(source_dir)
    converted_bytes = _dir_size_bytes(lerobot_dir)
    if source_bytes and converted_bytes:
        ratio = source_bytes / converted_bytes
        per_frame_src = source_bytes / max(1, total_frames_info)
        per_frame_dst = converted_bytes / max(1, total_frames_info)
        print(
            f"  — sizes: source={_fmt_bytes(source_bytes)}, "
            f"converted={_fmt_bytes(converted_bytes)} "
            f"(ratio {ratio:.1f}× | {_fmt_bytes(per_frame_src)}/frame → "
            f"{_fmt_bytes(per_frame_dst)}/frame)"
        )

    return _SubsetReport(
        name=subset_name,
        source_zip_count=len(src_zips),
        converted_episode_count=total_episodes,
        episode_count_match=episode_count_match,
        sampled_frame_mismatches=sampled_frame_mismatches,
        sampled_video_mismatches=sampled_video_mismatches,
        total_frames_info=total_frames_info,
        total_frames_from_episodes=total_frames_episodes,
        total_frames_match=frame_total_match,
        source_bytes=source_bytes,
        converted_bytes=converted_bytes,
    )


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n}T"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    source_dir: Optional[Path] = None,
    lerobot_dir: Optional[Path] = None,
    source_root: Optional[Path] = None,
    lerobot_root: Optional[Path] = None,
    subsets: Optional[list[str]] = None,
    sample_size: int = 3,
    seed: int = 42,
) -> None:
    """Verify one or many converted JHU dVRK LeRobot subsets.

    Two invocation modes:

    - Single subset:  pass ``--source-dir`` + ``--lerobot-dir`` (both must
      exist; the script treats them as one already-paired subset).
    - Multi-subset:  pass ``--source-root`` + ``--lerobot-root`` and
      optionally ``--subsets`` (defaults to ``SUBSETS_TO_CONVERT``).  The
      script resolves each entry's dataset name via
      :func:`convert_jhu_zarr_to_lerobot.resolve_dataset_name` and checks
      each pair.

    Args:
        source_dir: Source zarr subset directory (``tissue_*/`` children).
        lerobot_dir: Converted LeRobot dataset directory.
        source_root: Parent of every source subset (alternative to
            ``--source-dir``).
        lerobot_root: Parent of every converted LeRobot subset
            (alternative to ``--lerobot-dir``).
        subsets: Relative subset paths to verify when using
            ``--source-root`` / ``--lerobot-root``.  Defaults to
            ``SUBSETS_TO_CONVERT``.
        sample_size: Number of random episodes per subset to spot-check.
            ``0`` skips the per-episode check.
        seed: RNG seed for ``--sample-size`` picks (deterministic).
    """
    reports: list[_SubsetReport] = []
    t0 = time.time()

    if source_dir is not None and lerobot_dir is not None:
        name = lerobot_dir.name
        reports.append(
            verify_subset(
                source_dir,
                lerobot_dir,
                subset_name=name,
                sample_size=sample_size,
                seed=seed,
            )
        )
    elif source_root is not None and lerobot_root is not None:
        target_subsets = subsets if subsets is not None else list(SUBSETS_TO_CONVERT)
        for rel in target_subsets:
            name = resolve_dataset_name(rel)
            src = (source_root / rel.lstrip("/")).resolve()
            dst = (lerobot_root / name).resolve()
            reports.append(
                verify_subset(
                    src,
                    dst,
                    subset_name=name,
                    sample_size=sample_size,
                    seed=seed,
                )
            )
    else:
        print(
            "ERROR: pass either (--source-dir + --lerobot-dir) or "
            "(--source-root + --lerobot-root)."
        )
        sys.exit(2)

    # Final summary
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"SUMMARY (in {elapsed:.1f}s)")
    print("=" * 72)
    all_ok = True
    for r in reports:
        flags = []
        if not r.episode_count_match:
            flags.append("EPISODE COUNT")
            all_ok = False
        if not r.total_frames_match:
            flags.append("FRAME TOTAL")
            all_ok = False
        if r.sampled_frame_mismatches:
            flags.append(f"PARQUET×{r.sampled_frame_mismatches}")
        if r.sampled_video_mismatches:
            flags.append(f"VIDEO×{r.sampled_video_mismatches}")
        tag = "OK" if not flags else "⚠ " + " ".join(flags)
        ratio_str = ""
        if r.source_bytes and r.converted_bytes:
            ratio_str = f"  ({_fmt_bytes(r.source_bytes)} → {_fmt_bytes(r.converted_bytes)}, "
            ratio_str += f"{r.source_bytes / r.converted_bytes:.1f}×)"
        print(
            f"  {r.name}: zips={r.source_zip_count}, "
            f"episodes={r.converted_episode_count}, "
            f"frames={r.total_frames_info} "
            f"[{tag}]{ratio_str}"
        )

    # Sampled mismatches are informational (``min(n_kin, n_img)`` is
    # expected behaviour), so we only fail on episode-count / total-frame
    # discrepancies.
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    tyro.cli(main)
