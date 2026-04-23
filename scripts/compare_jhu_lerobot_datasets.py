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
Structurally compare newly-converted LeRobot subsets against a reference
dataset (e.g. the Open-H ``hf_suturebot``).

Goal: make sure our new subsets have the same *schema* as the reference so
they can be consumed by the same training pipeline.  We intentionally compare
only *structural* properties (feature dtypes/shapes, modality.json slice
layout, parquet column schema, video resolution+codec+fps) and call out
expected-to-differ values (episode counts, task labels, per-episode stats)
in a separate section.

Why not strict equality?
------------------------
The two datasets legitimately differ on many axes:

- Different episode counts (different source trees).
- Different task labels (different subtasks per subset).
- Different data distributions → different ``stats.json`` / ``episodes_stats``
  values.
- The reference may lack ``stats_cosmos.json`` (required at training time),
  and so may our new subsets; we report its absence for both rather than
  failing.

So this script uses a **diff-style** report: for each property we compare,
print the reference value on the left and the test value on the right, with a
``✓`` / ``⚠`` / ``❌`` marker.  ``❌`` is reserved for structural mismatches
the training pipeline cannot tolerate; ``⚠`` is a heads-up that deserves
attention; ``ℹ`` is informational.

Usage
-----
Compare every subset under an output root to a single reference::

    python scripts/compare_jhu_lerobot_datasets.py \
        --reference-dir /path/to/Open-H/.../hf_suturebot \
        --tests-root   /path/to/Open-H_failures_ood/.../previously_collected_data \
        --sample-size 2

Compare one specific dataset to the reference::

    python scripts/compare_jhu_lerobot_datasets.py \
        --reference-dir /path/to/hf_suturebot \
        --test-dir      /path/to/knot_tying
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import av  # PyAV is a LeRobot dependency and already present at runtime.
import pandas as pd
import tyro


# ---------------------------------------------------------------------------
# Metadata that distinguishes "expected to differ" from "must match"
# ---------------------------------------------------------------------------

# ``info.json`` keys whose *values* must match between datasets for them to be
# compatible with the same training pipeline.  Counts/splits are deliberately
# excluded — they always differ.
INFO_STRUCTURAL_KEYS: tuple[str, ...] = (
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "data_path",
    "video_path",
)

# ``info.json`` keys we print side-by-side but tolerate differences on
# because a divergence is expected.
INFO_CONTENT_KEYS: tuple[str, ...] = (
    "total_episodes",
    "total_frames",
    "total_tasks",
    "total_videos",
    "total_chunks",
    "splits",
)

# Structural fields of each feature entry that must match per-key.
FEATURE_STRUCTURAL_KEYS: tuple[str, ...] = ("dtype", "shape", "names")

# modality.json structural fields per state/action sub-key.
MODALITY_SA_STRUCTURAL_KEYS: tuple[str, ...] = (
    "start",
    "end",
    "rotation_type",
    "absolute",
    "dtype",
    "original_key",
)

# modality.json structural fields per video sub-key.
MODALITY_VIDEO_STRUCTURAL_KEYS: tuple[str, ...] = ("original_key",)


# ---------------------------------------------------------------------------
# Report data classes
# ---------------------------------------------------------------------------


@dataclass
class Diff:
    """A single difference encountered while comparing two values."""

    severity: str  # "OK" | "INFO" | "WARN" | "ERROR"
    path: str  # dotted key path, e.g. "info.features.observation.state.shape"
    ref_value: Any = None
    test_value: Any = None
    note: str = ""

    @property
    def glyph(self) -> str:
        return {"OK": "✓", "INFO": "ℹ", "WARN": "⚠", "ERROR": "❌"}[self.severity]


@dataclass
class DatasetReport:
    """Collected comparison result for one ref↔test dataset pair."""

    test_name: str
    structural: list[Diff] = field(default_factory=list)
    content: list[Diff] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for d in self.structural if d.severity == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for d in self.structural if d.severity == "WARN")


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_first(path: Path, n: int = 1) -> list[dict]:
    """Read up to ``n`` JSONL lines from ``path``.  Returns ``[]`` if missing."""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def _first_episode_files(dataset: Path, info: dict) -> tuple[Optional[Path], list[Path]]:
    """Locate the first episode's parquet and every matching MP4 under it.

    Uses the ``data_path`` / ``video_path`` templates in ``info`` when
    available; otherwise falls back to globbing.  The video path isn't
    unique — each camera gets its own MP4 under
    ``videos/chunk-XXX/<feature_key>/episode_000000.mp4`` — so we return the
    full list so the caller can compare multi-camera setups too.
    """
    data_tpl = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_tpl = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    parquet = dataset / data_tpl.format(episode_chunk=0, episode_index=0)
    if not parquet.exists():
        # Some tools use a slightly different template; glob as a fallback.
        candidates = sorted((dataset / "data").glob("chunk-*/episode_000000.parquet"))
        parquet = candidates[0] if candidates else None

    video_keys = _video_feature_keys(info)
    mp4s: list[Path] = []
    for vk in video_keys:
        vp = dataset / video_tpl.format(
            episode_chunk=0, episode_index=0, video_key=vk
        )
        if vp.exists():
            mp4s.append(vp)
    if not mp4s:
        mp4s = sorted((dataset / "videos").glob("chunk-*/*/episode_000000.mp4"))

    return parquet, mp4s


def _video_feature_keys(info: dict) -> list[str]:
    """Feature keys of type ``video`` in ``info.features``."""
    return sorted(
        k
        for k, v in (info.get("features") or {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    )


# ---------------------------------------------------------------------------
# fd-level stderr silencer (shared with verify_jhu_lerobot.py in spirit)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _suppress_fd2() -> "Iterator[None]":
    """Silence native writes to fd 2 (stderr) — used around PyAV ``av.open``
    so AV1 decoder init chatter doesn't pollute our report."""
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


# ---------------------------------------------------------------------------
# Per-file comparators
# ---------------------------------------------------------------------------


def _diff_scalar(
    path: str,
    ref_value: Any,
    test_value: Any,
    *,
    severity_if_diff: str = "ERROR",
) -> Diff:
    if ref_value == test_value:
        return Diff("OK", path, ref_value, test_value)
    return Diff(severity_if_diff, path, ref_value, test_value)


def _compare_info(ref_info: dict, test_info: dict) -> tuple[list[Diff], list[Diff]]:
    """Structural + content diffs for ``info.json``."""
    structural: list[Diff] = []
    content: list[Diff] = []

    for key in INFO_STRUCTURAL_KEYS:
        structural.append(
            _diff_scalar(
                f"info.{key}",
                ref_info.get(key, None),
                test_info.get(key, None),
                # Only ``codebase_version`` can reasonably drift without breaking
                # the pipeline — downgrade to WARN.  The rest are ERROR.
                severity_if_diff="WARN" if key == "codebase_version" else "ERROR",
            )
        )

    # Features (per-key structural comparison).
    ref_features = ref_info.get("features") or {}
    test_features = test_info.get("features") or {}
    feature_keys = sorted(set(ref_features) | set(test_features))
    for fk in feature_keys:
        rf = ref_features.get(fk)
        tf = test_features.get(fk)
        if rf is None:
            structural.append(
                Diff(
                    "INFO",
                    f"info.features.{fk}",
                    None,
                    tf,
                    note="present in test only",
                )
            )
            continue
        if tf is None:
            structural.append(
                Diff(
                    "INFO",
                    f"info.features.{fk}",
                    rf,
                    None,
                    note="present in reference only",
                )
            )
            continue
        for subkey in FEATURE_STRUCTURAL_KEYS:
            structural.append(
                _diff_scalar(
                    f"info.features.{fk}.{subkey}",
                    rf.get(subkey),
                    tf.get(subkey),
                )
            )

    # Content-only keys (expected to differ).
    for key in INFO_CONTENT_KEYS:
        content.append(
            _diff_scalar(
                f"info.{key}",
                ref_info.get(key),
                test_info.get(key),
                severity_if_diff="INFO",
            )
        )

    return structural, content


def _compare_modality(
    ref: Optional[dict], test: Optional[dict]
) -> list[Diff]:
    """Structural diff for ``meta/modality.json``.

    Returns ``[]`` if neither side has a modality.json; returns a single
    ``INFO`` diff if only one side has one.
    """
    if ref is None and test is None:
        return [Diff("INFO", "modality.json", None, None, note="absent in both")]
    if ref is None:
        return [Diff("INFO", "modality.json", None, "present", note="present in test only")]
    if test is None:
        return [Diff("INFO", "modality.json", "present", None, note="present in reference only")]

    out: list[Diff] = []

    # state + action subkeys.
    for top in ("state", "action"):
        ref_top = ref.get(top) or {}
        test_top = test.get(top) or {}
        subkeys = sorted(set(ref_top) | set(test_top))
        for sk in subkeys:
            rsk = ref_top.get(sk)
            tsk = test_top.get(sk)
            key_path = f"modality.{top}.{sk}"
            if rsk is None:
                out.append(Diff("INFO", key_path, None, tsk, note="test only"))
                continue
            if tsk is None:
                out.append(Diff("INFO", key_path, rsk, None, note="reference only"))
                continue
            for subfield in MODALITY_SA_STRUCTURAL_KEYS:
                out.append(
                    _diff_scalar(
                        f"{key_path}.{subfield}",
                        rsk.get(subfield),
                        tsk.get(subfield),
                        # WARN on rotation_type/absolute/dtype differences
                        # (might still be trainable); ERROR on index mismatch
                        # (slicing will be wrong).
                        severity_if_diff="ERROR" if subfield in ("start", "end") else "WARN",
                    )
                )

    # video subkeys.
    ref_video = ref.get("video") or {}
    test_video = test.get("video") or {}
    subkeys = sorted(set(ref_video) | set(test_video))
    for sk in subkeys:
        rsk = ref_video.get(sk)
        tsk = test_video.get(sk)
        key_path = f"modality.video.{sk}"
        if rsk is None:
            out.append(Diff("INFO", key_path, None, tsk, note="test only"))
            continue
        if tsk is None:
            out.append(Diff("INFO", key_path, rsk, None, note="reference only"))
            continue
        for subfield in MODALITY_VIDEO_STRUCTURAL_KEYS:
            out.append(
                _diff_scalar(
                    f"{key_path}.{subfield}",
                    rsk.get(subfield),
                    tsk.get(subfield),
                )
            )

    return out


def _parquet_schema(path: Path) -> tuple[dict[str, str], dict[str, tuple]]:
    """Inspect the first-episode parquet and return ``(column→dtype, column→sample_shape)``.

    Sample shape is computed from the first row of each column — useful for
    ``observation.state`` / ``action`` where dtype alone is ``object`` (list).
    """
    df = pd.read_parquet(path)
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    shapes: dict[str, tuple] = {}
    if len(df) > 0:
        for c in df.columns:
            v = df[c].iloc[0]
            try:
                import numpy as np  # local import — we only need it here

                arr = np.asarray(v)
                shapes[c] = tuple(arr.shape)
            except Exception:
                shapes[c] = ()
    return dtypes, shapes


def _compare_parquet(
    ref_path: Optional[Path], test_path: Optional[Path]
) -> list[Diff]:
    """Compare column set, dtypes, and element shapes of the first parquets."""
    out: list[Diff] = []
    if ref_path is None or not ref_path.exists():
        return [
            Diff(
                "ERROR",
                "data.episode_0.parquet",
                str(ref_path) if ref_path else None,
                None,
                note="reference parquet not found",
            )
        ]
    if test_path is None or not test_path.exists():
        return [
            Diff(
                "ERROR",
                "data.episode_0.parquet",
                None,
                str(test_path) if test_path else None,
                note="test parquet not found",
            )
        ]

    ref_dtypes, ref_shapes = _parquet_schema(ref_path)
    test_dtypes, test_shapes = _parquet_schema(test_path)
    all_cols = sorted(set(ref_dtypes) | set(test_dtypes))

    for col in all_cols:
        if col not in ref_dtypes:
            out.append(
                Diff("INFO", f"parquet.columns.{col}", None, test_dtypes[col], note="test only")
            )
            continue
        if col not in test_dtypes:
            out.append(
                Diff("INFO", f"parquet.columns.{col}", ref_dtypes[col], None, note="reference only")
            )
            continue
        out.append(_diff_scalar(f"parquet.dtype.{col}", ref_dtypes[col], test_dtypes[col]))
        if col in ref_shapes and col in test_shapes:
            out.append(
                _diff_scalar(
                    f"parquet.elem_shape.{col}",
                    ref_shapes[col],
                    test_shapes[col],
                    severity_if_diff="WARN",
                )
            )

    return out


def _video_properties(path: Path) -> dict[str, Any]:
    """Extract resolution / fps / codec / pix_fmt / nb_frames from an MP4 via PyAV."""
    props: dict[str, Any] = {"path": str(path)}
    try:
        with _suppress_fd2():
            with av.open(str(path)) as container:
                if not container.streams.video:
                    props["error"] = "no video stream"
                    return props
                stream = container.streams.video[0]
                props["codec"] = stream.codec_context.name
                props["width"] = stream.codec_context.width
                props["height"] = stream.codec_context.height
                props["pix_fmt"] = (
                    stream.codec_context.pix_fmt
                    if stream.codec_context.pix_fmt
                    else None
                )
                try:
                    # Average frame rate as Fraction.
                    if stream.average_rate is not None:
                        props["fps"] = float(stream.average_rate)
                    else:
                        props["fps"] = None
                except Exception:
                    props["fps"] = None
                props["nb_frames"] = int(getattr(stream, "frames", 0) or 0)
    except Exception as e:
        props["error"] = f"{type(e).__name__}: {e}"
    return props


def _compare_videos(
    ref_mp4s: list[Path], test_mp4s: list[Path]
) -> list[Diff]:
    """Compare the first MP4 from each side: codec/resolution/fps/pix_fmt."""
    if not ref_mp4s and not test_mp4s:
        return [Diff("INFO", "videos", None, None, note="no MP4 files in either dataset")]
    if not ref_mp4s:
        return [Diff("ERROR", "videos", None, str(test_mp4s[0]), note="reference has no MP4")]
    if not test_mp4s:
        return [Diff("ERROR", "videos", str(ref_mp4s[0]), None, note="test has no MP4")]

    # Pair by camera stem ('endoscope_left', 'wrist.left', ...) when possible.
    out: list[Diff] = []
    # Map feature-key-part → path (uses the parent dir name).
    def _by_camera(paths: list[Path]) -> dict[str, Path]:
        return {p.parent.name: p for p in paths}

    ref_map = _by_camera(ref_mp4s)
    test_map = _by_camera(test_mp4s)
    cameras = sorted(set(ref_map) | set(test_map))

    for cam in cameras:
        rp = ref_map.get(cam)
        tp = test_map.get(cam)
        key_path = f"videos.{cam}"
        if rp is None:
            out.append(Diff("INFO", key_path, None, str(tp), note="test only"))
            continue
        if tp is None:
            out.append(Diff("INFO", key_path, str(rp), None, note="reference only"))
            continue

        ref_props = _video_properties(rp)
        test_props = _video_properties(tp)
        for field_ in ("codec", "width", "height", "pix_fmt", "fps"):
            out.append(
                _diff_scalar(
                    f"{key_path}.{field_}",
                    ref_props.get(field_),
                    test_props.get(field_),
                    # fps can differ by small epsilon (Fraction rounding) —
                    # report as WARN rather than ERROR so a 30 vs 30.0 mismatch
                    # doesn't look catastrophic.
                    severity_if_diff="WARN" if field_ == "fps" else "ERROR",
                )
            )
    return out


def _file_presence(ref: Path, test: Path, meta_name: str) -> Diff:
    """One-line ``ℹ`` diff for the presence of an optional metadata file."""
    ref_has = (ref / "meta" / meta_name).exists()
    test_has = (test / "meta" / meta_name).exists()
    if ref_has and test_has:
        return Diff("OK", f"meta.{meta_name}", "present", "present")
    if not ref_has and not test_has:
        return Diff("INFO", f"meta.{meta_name}", "missing", "missing", note="absent in both")
    return Diff(
        "INFO",
        f"meta.{meta_name}",
        "present" if ref_has else "missing",
        "present" if test_has else "missing",
        note="present in only one side",
    )


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------


def compare_one(ref: Path, test: Path, test_name: str) -> DatasetReport:
    """Compare one ``test`` dataset against the ``ref`` reference dataset."""
    report = DatasetReport(test_name=test_name)

    ref_info = _load_json(ref / "meta/info.json")
    test_info = _load_json(test / "meta/info.json")
    info_structural, info_content = _compare_info(ref_info, test_info)
    report.structural.extend(info_structural)
    report.content.extend(info_content)

    ref_mod_path = ref / "meta/modality.json"
    test_mod_path = test / "meta/modality.json"
    ref_mod = _load_json(ref_mod_path) if ref_mod_path.exists() else None
    test_mod = _load_json(test_mod_path) if test_mod_path.exists() else None
    report.structural.extend(_compare_modality(ref_mod, test_mod))

    ref_parquet, ref_mp4s = _first_episode_files(ref, ref_info)
    test_parquet, test_mp4s = _first_episode_files(test, test_info)
    report.structural.extend(_compare_parquet(ref_parquet, test_parquet))
    report.structural.extend(_compare_videos(ref_mp4s, test_mp4s))

    # Optional metadata files.
    for meta_name in (
        "stats.json",
        "stats_cosmos.json",
        "stats_cosmos-44D.json",
        "README.md",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_stats.jsonl",
    ):
        report.content.append(_file_presence(ref, test, meta_name))

    # Tasks — just report counts / first few, don't diff text.
    ref_tasks = _load_jsonl_first(ref / "meta/tasks.jsonl", n=10_000)
    test_tasks = _load_jsonl_first(test / "meta/tasks.jsonl", n=10_000)
    report.content.append(
        Diff(
            "INFO",
            "tasks.count",
            len(ref_tasks),
            len(test_tasks),
        )
    )

    return report


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


_COLORS = {
    "OK": "\033[92m",
    "INFO": "\033[94m",
    "WARN": "\033[93m",
    "ERROR": "\033[91m",
    "RESET": "\033[0m",
}


def _fmt_value(v: Any) -> str:
    """Short repr of ``v`` that fits on one line."""
    if v is None:
        return "None"
    if isinstance(v, (str, int, float, bool)):
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + "..."
        return s
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"{type(v).__name__}[len={len(v)}]"
    s = repr(v)
    if len(s) > 80:
        s = s[:77] + "..."
    return s


def _print_diff_section(title: str, diffs: list[Diff], verbose: bool) -> None:
    print(f"\n  [{title}]")
    if not diffs:
        print("    (nothing to report)")
        return
    for d in diffs:
        if d.severity == "OK" and not verbose:
            continue
        color = _COLORS.get(d.severity, "")
        reset = _COLORS["RESET"]
        head = f"    {color}{d.glyph}{reset} {d.path}"
        if d.severity == "OK" and d.ref_value == d.test_value:
            # In verbose mode we still print OK lines, but keep them short.
            print(f"{head}  = {_fmt_value(d.ref_value)}")
        else:
            line = f"{head}  ref={_fmt_value(d.ref_value)}  test={_fmt_value(d.test_value)}"
            if d.note:
                line += f"  ({d.note})"
            print(line)


def _print_report(report: DatasetReport, verbose: bool) -> None:
    if report.n_errors > 0:
        label = f"{_COLORS['ERROR']}STRUCTURAL MISMATCH ({report.n_errors}){_COLORS['RESET']}"
    elif report.n_warnings > 0:
        label = f"{_COLORS['WARN']}structural warnings ({report.n_warnings}){_COLORS['RESET']}"
    else:
        label = f"{_COLORS['OK']}structural match{_COLORS['RESET']}"

    print(f"\n=== {report.test_name} — {label} ===")

    _print_diff_section("structural", report.structural, verbose)
    _print_diff_section("content (expected to differ)", report.content, verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    reference_dir: Path,
    test_dir: Optional[Path] = None,
    tests_root: Optional[Path] = None,
    tests: Optional[list[str]] = None,
    verbose: bool = False,
) -> None:
    """Compare newly-converted LeRobot subsets to a reference dataset.

    Exactly one of ``--test-dir`` or ``--tests-root`` is required.

    Args:
        reference_dir: Path to the reference LeRobot dataset (e.g.
            ``.../hf_suturebot``).  Treated as the source of truth for the
            schema; we report everything as ``ref vs test``.
        test_dir: Single converted subset to compare against the reference.
            Mutually exclusive with ``--tests-root``.
        tests_root: Parent directory containing one subdirectory per
            converted subset.  Mutually exclusive with ``--test-dir``.
        tests: When ``--tests-root`` is given, the subset names to compare.
            Defaults to *every* direct subdirectory of ``--tests-root``.
        verbose: Also print ``OK`` entries (matching fields) — by default we
            only show non-matching rows to keep the report compact.
    """
    if not reference_dir.exists():
        print(f"ERROR: reference dir does not exist: {reference_dir}")
        sys.exit(2)

    if (test_dir is None) == (tests_root is None):
        print("ERROR: pass exactly one of --test-dir or --tests-root.")
        sys.exit(2)

    t0 = time.time()

    test_pairs: list[tuple[str, Path]] = []
    if test_dir is not None:
        test_pairs.append((test_dir.name, test_dir))
    else:
        assert tests_root is not None
        candidates = [p for p in sorted(tests_root.iterdir()) if p.is_dir()]
        if tests is not None:
            wanted = set(tests)
            candidates = [p for p in candidates if p.name in wanted]
            missing = wanted - {p.name for p in candidates}
            for m in sorted(missing):
                print(f"WARNING: --tests {m!r} not found under {tests_root}")
        for p in candidates:
            test_pairs.append((p.name, p))

    print(f"Reference: {reference_dir}")
    if tests_root is not None:
        print(f"Tests root: {tests_root}  ({len(test_pairs)} datasets)")
    print()

    reports: list[DatasetReport] = []
    for test_name, test_path in test_pairs:
        try:
            report = compare_one(reference_dir, test_path, test_name)
        except FileNotFoundError as e:
            print(f"\n=== {test_name} — ❌ LOAD FAILED ===")
            print(f"  {e}")
            continue
        _print_report(report, verbose=verbose)
        reports.append(report)

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"SUMMARY (in {elapsed:.1f}s)")
    print("=" * 72)
    ok_count = 0
    warn_count = 0
    err_count = 0
    for r in reports:
        if r.n_errors:
            status = f"{_COLORS['ERROR']}ERRORS={r.n_errors}{_COLORS['RESET']}"
            err_count += 1
        elif r.n_warnings:
            status = f"{_COLORS['WARN']}WARN={r.n_warnings}{_COLORS['RESET']}"
            warn_count += 1
        else:
            status = f"{_COLORS['OK']}OK{_COLORS['RESET']}"
            ok_count += 1
        print(f"  {r.test_name}: {status}")

    print()
    print(
        f"  totals: OK={ok_count}  warnings={warn_count}  errors={err_count}  "
        f"of {len(reports)} datasets"
    )
    sys.exit(0 if err_count == 0 else 1)


if __name__ == "__main__":
    tyro.cli(main)
