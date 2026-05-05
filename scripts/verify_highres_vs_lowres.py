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
Verify that the high-resolution (720 H x 960 W) JHU dVRK LeRobot rebuild
contains the **same content** as the existing 540 H x 960 W reference
datasets — i.e. nothing was lost in the re-conversion modulo the planned
resolution change.

Scope
-----
The high-res converter (``convert_jhu_zarr_to_lerobot_highres.py``)
produced 9 LeRobot datasets under one parent root (e.g.
``.../JHU_data_jpeg100_noacc_clean++/LeRobot_720x960/``).  Each subset has
a pre-existing 540x960 counterpart that we trust as the source of truth:

- 8 subsets (``knot_tying``, ``suture_bot_success``, ``suture_bot_failure``,
  ``ood``, ``cosmos_fail_filtered``, ``cosmos_throw_fail_demo``,
  ``cosmos_knot_fail_demo``, ``suturebot_act_throw_eval``) live under
  ``Open-H_failures_ood/Surgical/JHU/Imerse/previously_collected_data/``.
- ``hf_suturebot`` lives under
  ``Open-H/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot``.

What is checked
---------------
For each subset we compare the existing 540x960 reference (LOWRES) with
the new 720x960 build (HIGHRES) on:

* **total_episodes** (strict — must match)
* **total_frames** (strict)
* **per-episode length multiset** (strict — same set of episode lengths,
  regardless of how the two converters happened to order episodes)
* **task name set** (strict — same set of natural-language task labels)
* **fps** (strict)
* **disk artifacts** — parquet file count, MP4 file count, both verified
  against the totals declared in ``info.json``.
* **video resolution** (informational — expected to differ:
  ``[540, 960, 3]`` vs ``[720, 960, 3]``).
* **camera count** (informational — expected to differ for
  ``hf_suturebot``: the existing reference has 4 cameras
  (endoscope.left/right + wrist.left/right) while our high-res build
  keeps only the left endoscope, matching the ``jhu_dvrk_mono``
  registry).
* **robot_type** (informational — expected to differ for ``hf_suturebot``
  for the same reason: ``"dvrk"`` vs ``"jhu_dvrk_mono"``).

Schema-level differences that we already analyzed in
``compare_jhu_lerobot_datasets.py`` (nested ``names`` lists, dotted vs
underscored camera keys, optional modality fields, axis-name synonyms,
etc.) are intentionally **not** rechecked here — that's a separate
script's job.  The focus of this verifier is purely "did the same data
make it across at the new resolution".

Usage
-----
Default cluster paths::

    python scripts/verify_highres_vs_lowres.py \\
        --highres-root /lustre/fsw/.../JHU_data_jpeg100_noacc_clean++/LeRobot_720x960

Override the lowres locations if your mounts differ::

    python scripts/verify_highres_vs_lowres.py \\
        --highres-root /lustre/.../LeRobot_720x960 \\
        --lowres-failures-ood-root /lustre/.../Open-H_failures_ood/.../previously_collected_data \\
        --lowres-hf-suturebot-dir /lustre/.../hf_suturebot

Limit to one subset::

    python scripts/verify_highres_vs_lowres.py \\
        --highres-root .../LeRobot_720x960 \\
        --subsets hf_suturebot
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tyro


# ---------------------------------------------------------------------------
# Default reference paths (cluster layout)
# ---------------------------------------------------------------------------

# 8 subsets that share a common parent directory (the "failures + OOD" pile
# we converted from raw zarr).
DEFAULT_LOWRES_FAILURES_OOD_ROOT = Path(
    "/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/Open-H_failures_ood/Surgical/JHU/Imerse/previously_collected_data"
)
SUBSETS_FROM_FAILURES_OOD: tuple[str, ...] = (
    "knot_tying",
    "suture_bot_success",
    "suture_bot_failure",
    "ood",
    "cosmos_fail_filtered",
    "cosmos_throw_fail_demo",
    "cosmos_knot_fail_demo",
    "suturebot_act_throw_eval",
)

# The 9th subset ``hf_suturebot`` lives in a different directory tree.
DEFAULT_LOWRES_HF_SUTUREBOT_DIR = Path(
    "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/Open-H/Surgical/JHU/Imerse/previously_collected_data/hf_suturebot"
)
SUBSET_HF_SUTUREBOT: str = "hf_suturebot"

ALL_SUBSETS: tuple[str, ...] = SUBSETS_FROM_FAILURES_OOD + (SUBSET_HF_SUTUREBOT,)


# ---------------------------------------------------------------------------
# Per-side summary
# ---------------------------------------------------------------------------


@dataclass
class Summary:
    """Compact extract of a LeRobot dataset's content invariants.

    All fields here are values that should match (or be explicitly
    expected to differ) between the lowres and highres versions of the
    same logical dataset.  Resolution-only differences are kept on the
    side (``video_shape``, ``n_cameras``) and reported as informational.
    """

    name: str
    path: Path

    # info.json totals
    total_episodes: int
    total_frames: int
    total_videos: int
    total_tasks: int
    total_chunks: int
    chunks_size: int
    fps: int
    robot_type: str

    # Derived from features
    video_shape: list[int]  # [H, W, C] of the first video feature
    n_cameras: int

    # Per-episode lengths from episodes.jsonl (used as a multiset)
    episode_lengths: list[int]

    # Set of natural-language task labels
    task_names: set[str]

    # On-disk artifact counts (used to cross-check info.json)
    n_parquet_files: int
    n_mp4_files: int


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize(dataset_path: Path, subset_name: str) -> Optional[Summary]:
    """Build a :class:`Summary` for one LeRobot dataset directory.

    Returns ``None`` if the dataset clearly isn't present (missing
    ``meta/info.json``).  All other parsing errors propagate so they can
    be reported as failures.
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"[{subset_name}] ❌ missing meta/info.json under {dataset_path}")
        return None

    with info_path.open("r", encoding="utf-8") as f:
        info: dict[str, Any] = json.load(f)

    episodes = _load_jsonl(dataset_path / "meta" / "episodes.jsonl")
    tasks = _load_jsonl(dataset_path / "meta" / "tasks.jsonl")

    # Identify all video features and pick the first as representative
    # (resolution must be uniform across video features in v2.1, so this
    # is enough for our INFO line about resolution).
    features = info.get("features") or {}
    video_features = [
        k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    video_shape: list[int] = []
    if video_features:
        f0 = features[video_features[0]]
        video_shape = list(f0.get("shape") or [])

    # Disk-level counts.  Cheap because each subset has at most a few
    # thousand parquets and a few thousand MP4s on lustre.
    n_parquets = sum(1 for _ in dataset_path.glob("data/chunk-*/episode_*.parquet"))
    n_mp4s = sum(1 for _ in dataset_path.glob("videos/chunk-*/*/episode_*.mp4"))

    return Summary(
        name=subset_name,
        path=dataset_path,
        total_episodes=int(info.get("total_episodes", 0)),
        total_frames=int(info.get("total_frames", 0)),
        total_videos=int(info.get("total_videos", 0)),
        total_tasks=int(info.get("total_tasks", 0)),
        total_chunks=int(info.get("total_chunks", 0)),
        chunks_size=int(info.get("chunks_size", 1000)),
        fps=int(info.get("fps", 0)),
        robot_type=str(info.get("robot_type", "")),
        video_shape=video_shape,
        n_cameras=len(video_features),
        episode_lengths=[int(e.get("length", 0)) for e in episodes],
        task_names={str(t.get("task", "")) for t in tasks},
        n_parquet_files=n_parquets,
        n_mp4_files=n_mp4s,
    )


# ---------------------------------------------------------------------------
# Diff records and printing
# ---------------------------------------------------------------------------


@dataclass
class Diff:
    """One comparison line between LOWRES and HIGHRES for a given subset."""

    severity: str  # "OK" | "INFO" | "WARN" | "ERROR"
    field: str
    lowres: Any
    highres: Any
    note: str = ""

    @property
    def glyph(self) -> str:
        return {"OK": "✓", "INFO": "ℹ", "WARN": "⚠", "ERROR": "❌"}[self.severity]


@dataclass
class SubsetReport:
    name: str
    diffs: list[Diff] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for d in self.diffs if d.severity == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for d in self.diffs if d.severity == "WARN")


_COLORS = {
    "OK": "\033[92m",
    "INFO": "\033[94m",
    "WARN": "\033[93m",
    "ERROR": "\033[91m",
    "RESET": "\033[0m",
}


def _fmt(v: Any) -> str:
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "..."


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _diff_strict(field_: str, lo: Any, hi: Any) -> Diff:
    return (
        Diff("OK", field_, lo, hi)
        if lo == hi
        else Diff("ERROR", field_, lo, hi)
    )


def _diff_info(field_: str, lo: Any, hi: Any, note: str = "") -> Diff:
    return Diff("INFO", field_, lo, hi, note)


def compare(lo: Summary, hi: Summary) -> SubsetReport:
    """Compare two summaries and produce a list of typed diffs.

    Strict (``ERROR`` on mismatch):
        total_episodes, total_frames, fps, episode_lengths_multiset,
        task_names, parquet_count_consistency, mp4_count_consistency.

    Informational (``INFO``, mismatch is expected/allowed):
        video_shape (resolution change is the whole point), n_cameras
        (only ``hf_suturebot`` differs: 4 vs 1), robot_type
        (``dvrk`` vs ``jhu_dvrk_mono`` for ``hf_suturebot``),
        total_videos (depends on n_cameras, can be derived from totals),
        total_chunks, chunks_size.
    """
    r = SubsetReport(name=lo.name)

    # --- core content invariants (strict) ---
    r.diffs.append(_diff_strict("total_episodes", lo.total_episodes, hi.total_episodes))
    r.diffs.append(_diff_strict("total_frames", lo.total_frames, hi.total_frames))
    r.diffs.append(_diff_strict("fps", lo.fps, hi.fps))

    # Episode-length multiset (independent of episode ordering — the lowres
    # ``hf_suturebot`` reference came out of a different converter that may
    # have walked the source roots in a different order, so ``episode 0`` on
    # one side is not necessarily ``episode 0`` on the other.  Comparing the
    # multiset of lengths captures the content invariant we care about
    # ("same set of episodes, each of the same length") without depending on
    # ordering).
    lo_ml = Counter(lo.episode_lengths)
    hi_ml = Counter(hi.episode_lengths)
    if lo_ml == hi_ml:
        r.diffs.append(
            Diff("OK", "episode_lengths_multiset",
                 f"{len(lo.episode_lengths)} eps", f"{len(hi.episode_lengths)} eps")
        )
    else:
        only_lo = lo_ml - hi_ml
        only_hi = hi_ml - lo_ml
        r.diffs.append(
            Diff(
                "ERROR",
                "episode_lengths_multiset",
                f"{sum(only_lo.values())} unmatched lengths",
                f"{sum(only_hi.values())} unmatched lengths",
                note=(
                    f"lowres-only {dict(sorted(only_lo.items()))}; "
                    f"highres-only {dict(sorted(only_hi.items()))}"
                ),
            )
        )

    # Task name set
    if lo.task_names == hi.task_names:
        r.diffs.append(
            Diff("OK", "task_names",
                 f"{len(lo.task_names)} tasks", f"{len(hi.task_names)} tasks")
        )
    else:
        r.diffs.append(
            Diff(
                "ERROR",
                "task_names",
                _fmt(sorted(lo.task_names)),
                _fmt(sorted(hi.task_names)),
                note=(
                    f"lowres-only={sorted(lo.task_names - hi.task_names)}; "
                    f"highres-only={sorted(hi.task_names - lo.task_names)}"
                ),
            )
        )

    # --- on-disk consistency (strict) ---
    # parquet count must equal total_episodes on each side
    if lo.n_parquet_files != lo.total_episodes:
        r.diffs.append(Diff(
            "ERROR", "lowres.parquet_count_vs_info",
            lo.n_parquet_files, lo.total_episodes,
            note="parquet count on disk != info.total_episodes"
        ))
    if hi.n_parquet_files != hi.total_episodes:
        r.diffs.append(Diff(
            "ERROR", "highres.parquet_count_vs_info",
            hi.n_parquet_files, hi.total_episodes,
            note="parquet count on disk != info.total_episodes"
        ))
    if lo.n_parquet_files == lo.total_episodes and hi.n_parquet_files == hi.total_episodes:
        r.diffs.append(Diff(
            "OK", "parquet_files",
            f"{lo.n_parquet_files} on disk", f"{hi.n_parquet_files} on disk"
        ))

    # mp4 count = total_videos = n_cameras × total_episodes
    expected_lo_mp4s = lo.n_cameras * lo.total_episodes
    expected_hi_mp4s = hi.n_cameras * hi.total_episodes
    if lo.n_mp4_files != expected_lo_mp4s:
        r.diffs.append(Diff(
            "ERROR", "lowres.mp4_count_vs_info",
            lo.n_mp4_files, expected_lo_mp4s,
            note=f"on-disk MP4 count != n_cameras ({lo.n_cameras}) × total_episodes"
        ))
    if hi.n_mp4_files != expected_hi_mp4s:
        r.diffs.append(Diff(
            "ERROR", "highres.mp4_count_vs_info",
            hi.n_mp4_files, expected_hi_mp4s,
            note=f"on-disk MP4 count != n_cameras ({hi.n_cameras}) × total_episodes"
        ))
    if lo.n_mp4_files == expected_lo_mp4s and hi.n_mp4_files == expected_hi_mp4s:
        r.diffs.append(Diff(
            "OK", "mp4_files",
            f"{lo.n_mp4_files} ({lo.n_cameras} cam × {lo.total_episodes} eps)",
            f"{hi.n_mp4_files} ({hi.n_cameras} cam × {hi.total_episodes} eps)",
        ))

    # --- expected-to-differ (info) ---
    r.diffs.append(_diff_info(
        "video_shape", lo.video_shape, hi.video_shape,
        note="resolution change is the whole point of the high-res rebuild"
    ))
    if lo.n_cameras != hi.n_cameras:
        r.diffs.append(Diff(
            "INFO", "n_cameras",
            lo.n_cameras, hi.n_cameras,
            note=("expected for hf_suturebot: lowres ref has 4 cams "
                  "(endoscope.left+right + wrist.left+right); highres "
                  "build keeps only the left endoscope per "
                  "EMBODIMENT_REGISTRY['jhu_dvrk_mono']")
        ))
    else:
        r.diffs.append(_diff_info("n_cameras", lo.n_cameras, hi.n_cameras))

    # total_videos = n_cameras × total_episodes; differs whenever n_cameras differs
    r.diffs.append(_diff_info("total_videos", lo.total_videos, hi.total_videos))
    r.diffs.append(_diff_info("robot_type", lo.robot_type, hi.robot_type))

    # total_tasks may differ if the lowres version had recovery suffixes
    # parsed differently — informational unless task_names already errored.
    if lo.total_tasks == hi.total_tasks:
        r.diffs.append(Diff("OK", "total_tasks", lo.total_tasks, hi.total_tasks))
    else:
        r.diffs.append(Diff(
            "WARN", "total_tasks", lo.total_tasks, hi.total_tasks,
            note="task counts differ — see task_names diff for details"
        ))

    return r


# ---------------------------------------------------------------------------
# Per-subset entry point
# ---------------------------------------------------------------------------


def _resolve_lowres_path(
    subset_name: str,
    lowres_failures_ood_root: Path,
    lowres_hf_suturebot_dir: Path,
) -> Path:
    if subset_name == SUBSET_HF_SUTUREBOT:
        return lowres_hf_suturebot_dir
    return lowres_failures_ood_root / subset_name


def verify_one(
    subset_name: str,
    *,
    lowres_path: Path,
    highres_path: Path,
    verbose: bool,
) -> SubsetReport:
    """Verify a single (lowres, highres) pair and pretty-print the diff."""
    print(f"\n=== {subset_name} ===")
    print(f"  lowres : {lowres_path}")
    print(f"  highres: {highres_path}")

    lo = summarize(lowres_path, subset_name)
    hi = summarize(highres_path, subset_name)
    if lo is None or hi is None:
        # Already logged by summarize().  Return an empty report flagged
        # as a single ERROR so the run summary picks it up.
        report = SubsetReport(name=subset_name)
        report.diffs.append(
            Diff("ERROR", "load",
                 "missing" if lo is None else "ok",
                 "missing" if hi is None else "ok",
                 note="meta/info.json could not be read on at least one side")
        )
        for d in report.diffs:
            _print_diff(d)
        return report

    report = compare(lo, hi)
    for d in report.diffs:
        if d.severity == "OK" and not verbose:
            continue
        _print_diff(d)

    return report


def _print_diff(d: Diff) -> None:
    color = _COLORS.get(d.severity, "")
    reset = _COLORS["RESET"]
    head = f"  {color}{d.glyph}{reset} {d.field}"
    line = f"{head}  lowres={_fmt(d.lowres)}  highres={_fmt(d.highres)}"
    if d.note:
        line += f"  ({d.note})"
    print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    highres_root: Path,
    lowres_failures_ood_root: Path = DEFAULT_LOWRES_FAILURES_OOD_ROOT,
    lowres_hf_suturebot_dir: Path = DEFAULT_LOWRES_HF_SUTUREBOT_DIR,
    subsets: list[str] = list(ALL_SUBSETS),
    verbose: bool = False,
) -> None:
    """Verify the high-res rebuild's content against the existing 540x960
    reference datasets.

    Args:
        highres_root: Parent directory containing the 9 high-res LeRobot
            datasets (one subdirectory per subset, e.g.
            ``LeRobot_720x960/knot_tying/``).
        lowres_failures_ood_root: Parent directory of the 8 lowres
            ``Open-H_failures_ood`` subsets (the ones our converter
            originally produced at 540x960).
        lowres_hf_suturebot_dir: Path to the existing ``hf_suturebot``
            LeRobot dataset (the 9th subset, in a different parent dir
            from the other 8).
        subsets: Subset names to verify.  Defaults to all 9.
        verbose: Print ``OK`` lines too, not just diffs/info.
    """
    if not highres_root.exists():
        print(f"ERROR: --highres-root does not exist: {highres_root}")
        sys.exit(2)

    # Validate subset names early.
    invalid = [s for s in subsets if s not in ALL_SUBSETS]
    if invalid:
        print(f"ERROR: unknown subset(s): {invalid}; choose from {list(ALL_SUBSETS)}")
        sys.exit(2)

    print(f"Highres root: {highres_root}")
    print(f"Lowres failures+OOD root: {lowres_failures_ood_root}")
    print(f"Lowres hf_suturebot dir:  {lowres_hf_suturebot_dir}")
    print(f"Subsets ({len(subsets)}): {subsets}")

    reports: list[SubsetReport] = []
    t0 = time.time()
    for name in subsets:
        lowres_path = _resolve_lowres_path(
            name, lowres_failures_ood_root, lowres_hf_suturebot_dir
        )
        highres_path = highres_root / name
        report = verify_one(
            name,
            lowres_path=lowres_path,
            highres_path=highres_path,
            verbose=verbose,
        )
        reports.append(report)
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"SUMMARY (in {elapsed:.1f}s)")
    print("=" * 72)
    n_total_ok = 0
    for r in reports:
        if r.n_errors:
            status = f"{_COLORS['ERROR']}ERRORS={r.n_errors}{_COLORS['RESET']}"
        elif r.n_warnings:
            status = f"{_COLORS['WARN']}WARN={r.n_warnings}{_COLORS['RESET']}"
        else:
            status = f"{_COLORS['OK']}OK{_COLORS['RESET']}"
            n_total_ok += 1
        print(f"  {r.name}: {status}")

    n_with_errors = sum(1 for r in reports if r.n_errors)
    print()
    print(f"  totals: OK={n_total_ok}  with-errors={n_with_errors}  "
          f"of {len(reports)} datasets")
    sys.exit(0 if n_with_errors == 0 else 1)


if __name__ == "__main__":
    tyro.cli(main)
