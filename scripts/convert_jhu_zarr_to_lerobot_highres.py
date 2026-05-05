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
High-resolution (720 H x 960 W) variant of ``convert_jhu_zarr_to_lerobot.py``.

This is the "v2" converter we use to produce the JHU dVRK LeRobot datasets
that feed a Cosmos-H-Surgical-Simulator fine-tune at a resolution close to
the Cosmos-Predict2.5 base-model pre-training resolution
(``video_size=(704, 1280)`` HxW, see
``cosmos-predict2.5-1.5.1/cosmos_predict2/experiments/base/cosmos_nemo_assets_lora.py``).

What's different from the lower-res sibling
-------------------------------------------
1. **Target resolution**: 720 H x 960 W.  Source ``left`` frames are
   540 H x 960 W (16:9), so the existing aspect-preserving letterbox
   helper pads to 720 H x 960 W with 90 px black bars top + 90 px bottom.
   The 16:9 endoscope content stays undistorted.

2. **9th subset: ``hf_suturebot``.**  The existing ``hf_suturebot`` LeRobot
   dataset (the reference ``dvrk`` set under
   ``Open-H/Surgical/JHU/Imerse/previously_collected_data/``) was produced
   by ``open-h-embodiment/scripts/conversion/dvrk_zarr_to_lerobot.py`` from
   FOUR zarr source roots and then filtered down to **1452 episodes**
   (the union of all four roots is 1552; 100 ``2_needle_throw[_recovery]``
   episodes from ``tissue_1`` and ``tissue_2`` were dropped).  See
   ``agent_chats/hf_suturebot_coverage_analysis.md``.

   Because the user asked for the high-res rebuild to be **identical in
   episode scope** to that existing reference, we re-create the same
   1452-episode dataset by:

   a) walking all four zarr roots,
   b) filtering each candidate zip by its ``YYYYMMDD-HHMMSS-UUUUUU``
      timestamp ID against the whitelist extracted from the reference
      dataset's ``README_duplicates.md``.

How the resolution override works
---------------------------------
Rather than duplicate 1800+ lines of converter code, this script imports
the existing ``convert_jhu_zarr_to_lerobot`` module and *reassigns*
``TARGET_HEIGHT`` and ``TARGET_WIDTH`` on it.  Every helper function in
that module looks up those constants via its ``__globals__`` dict at call
time, so reassigning the module attributes propagates automatically to
``maybe_resize_frames``, ``_features_spec``, ``_empty_payload``,
``_decode_episode``, ``write_readme``, etc.  The patch must run *before*
any of those functions are invoked — we do it at module import time, on
line 92 below, before importing any individual symbol from the base
module.

Usage
-----
Convert every subset (8 + ``hf_suturebot``)::

    python scripts/convert_jhu_zarr_to_lerobot_highres.py \\
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \\
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono_720x960 \\
        --num-workers 16

Convert only ``hf_suturebot``, with a custom whitelist path::

    python scripts/convert_jhu_zarr_to_lerobot_highres.py \\
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \\
        --output-root $HF_LEROBOT_HOME/jhu_dvrk_mono_720x960 \\
        --subsets hf_suturebot \\
        --hf-suturebot-readme /lustre/.../hf_suturebot/README_duplicates.md

Convert a specific subset of the 8 base ones (sequential, debug)::

    python scripts/convert_jhu_zarr_to_lerobot_highres.py \\
        --input-base /lustre/.../JHU_data_jpeg100_noacc_clean++ \\
        --output-root /tmp/jhu_highres_debug \\
        --subsets knot_tying \\
        --num-workers 1 --max-episodes 2
"""

from __future__ import annotations

import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

# ---------------------------------------------------------------------------
# Resolution override (applies BEFORE any base helper is called).
# ---------------------------------------------------------------------------
# We must import the module first, patch its TARGET_* attributes, and ONLY
# THEN import the symbols we want to use locally.  Doing it this way (instead
# of ``from convert_jhu_zarr_to_lerobot import maybe_resize_frames`` followed
# by an attribute set on the local name) is what guarantees the patched
# values reach the inside of every helper, because Python functions look
# up module-level constants via their *defining* module's ``__globals__``,
# not via the import alias in this file.

# Locate the existing converter as a sibling module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import convert_jhu_zarr_to_lerobot as _base  # noqa: E402

# Patch resolution constants on the base module.  Every downstream lookup
# (resize, _features_spec, _empty_payload, README writer) sees these.
_base.TARGET_HEIGHT = 720
_base.TARGET_WIDTH = 960

# Local mirrors of the patched constants (for use inside this script's own
# code paths and for the README banner below).
TARGET_HEIGHT: int = _base.TARGET_HEIGHT
TARGET_WIDTH: int = _base.TARGET_WIDTH

# Re-import names AFTER the patch.  The names themselves don't capture the
# constants — the functions they refer to look them up at call time — but
# importing here keeps the rest of this file readable.
EpisodeRef = _base.EpisodeRef
convert_subset = _base.convert_subset
default_num_workers = _base.default_num_workers
JHU_PREFIX_DEFAULT = _base.JHU_PREFIX_DEFAULT


# ---------------------------------------------------------------------------
# Subset configuration
# ---------------------------------------------------------------------------

# Default cluster path of ``hf_suturebot/README_duplicates.md`` — the source
# of truth for the 1452 episode timestamp IDs that the reference
# ``hf_suturebot`` LeRobot dataset spans.  Override at runtime via
# ``--hf-suturebot-readme`` if your mount layout is different.
HF_SUTUREBOT_README_DEFAULT: str = (
    "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/Open-H/Surgical/JHU/Imerse/previously_collected_data/"
    "hf_suturebot/README_duplicates.md"
)

# JHU dVRK timestamp-ID regex.  Episode zips are named
# ``<YYYYMMDD-HHMMSS-UUUUUU>.zip`` (with optional ``_recovery`` suffix on
# the filename's stem).  The timestamp ID is what links a raw zarr zip to
# an entry in any LeRobot dataset built from it.
_TS_ID_RE: re.Pattern[str] = re.compile(r"(\d{8}-\d{6}-\d{6})")


@dataclass(frozen=True)
class SubsetConfig:
    """Output-dataset spec for the high-res rebuild.

    Attributes
    ----------
    name :
        Output dataset directory name (created under ``--output-root``).
    source_subpaths :
        List of paths *relative to* ``--input-base`` whose episodes are
        merged under this single output dataset.  Each entry may point at
        either a ``tissue_<N>`` directory directly, or at a parent
        directory whose immediate children are ``tissue_<N>`` dirs — both
        layouts are detected automatically.
    whitelist_md_path :
        Optional absolute path to a Markdown file (typically
        ``README_duplicates.md``) from which we extract a set of
        timestamp IDs (``\\d{8}-\\d{6}-\\d{6}``).  When set, only episode
        zips whose stem matches one of those IDs are included.  Used for
        ``hf_suturebot`` to exactly reproduce the 1452-episode scope of
        the reference LeRobot dataset.
    """

    name: str
    source_subpaths: tuple[str, ...]
    whitelist_md_path: Optional[str] = None


# All 9 subsets the high-res FT consumes.  The first 8 are 1:1 with the
# original ``SUBSETS_TO_CONVERT`` in the lower-res converter; the 9th
# (``hf_suturebot``) is new — see module docstring for sourcing rationale.
SUBSETS_TO_CONVERT_HIGHRES: tuple[SubsetConfig, ...] = (
    SubsetConfig("knot_tying", ("/knot_tying",)),
    SubsetConfig(
        "suture_bot_success", ("/suture_bot/success/processed_data_zipped_pi",)
    ),
    SubsetConfig(
        "suture_bot_failure", ("/suture_bot/failure/processed_data_zipped_pi",)
    ),
    SubsetConfig("ood", ("/ood/processed_data_zipped_pi",)),
    SubsetConfig("cosmos_fail_filtered", ("/cosmos_fail_filtered",)),
    SubsetConfig("cosmos_throw_fail_demo", ("/cosmos_throw_fail_demo",)),
    SubsetConfig("cosmos_knot_fail_demo", ("/cosmos_knot_fail_demo",)),
    SubsetConfig("suturebot_act_throw_eval", ("/suturebot_act_throw_eval",)),
    SubsetConfig(
        "hf_suturebot",
        (
            # Four zarr roots that together contain the 1552-episode super-set
            # of the reference ``hf_suturebot`` LeRobot dataset.  See
            # ``agent_chats/hf_suturebot_coverage_analysis.md`` for the full
            # provenance breakdown.  Note: the first two entries point at a
            # specific ``tissue_<N>`` directory (depth N+1 below the JHU
            # prefix); the last two point at the directory *above* tissue,
            # so both layouts must be supported by the discovery walker.
            "/suturing/Jesse/processed_data_zipped_pi/tissue_1",
            "/suturing_2/processed_data_zipped_pi/tissue_2",
            "/suturing_2/processed_suturing_data_zipped_pi",
            "/suturing_3/processed_suturing_data_zipped_pi_clean",
        ),
        whitelist_md_path=HF_SUTUREBOT_README_DEFAULT,
    ),
)


def _subset_by_name(name: str) -> SubsetConfig:
    for s in SUBSETS_TO_CONVERT_HIGHRES:
        if s.name == name:
            return s
    raise KeyError(
        f"Unknown subset name {name!r}; choose from "
        f"{[s.name for s in SUBSETS_TO_CONVERT_HIGHRES]}"
    )


# ---------------------------------------------------------------------------
# Episode discovery (multi-root + ts-id whitelist)
# ---------------------------------------------------------------------------


def _is_tissue_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith("tissue_")


def _walk_tissue_dirs(root: Path) -> list[Path]:
    """Return every ``tissue_*`` directory at-or-just-below ``root``.

    If ``root`` itself is a ``tissue_*`` directory, returns ``[root]``.
    Otherwise returns the immediate ``tissue_*`` children of ``root``,
    sorted by name.  Returns ``[]`` if ``root`` doesn't exist (we log
    that case in the caller).
    """
    if not root.exists():
        return []
    if _is_tissue_dir(root):
        return [root]
    return sorted(p for p in root.iterdir() if _is_tissue_dir(p))


def _collect_tissue_episodes(tissue_dir: Path) -> list[EpisodeRef]:
    """Walk one tissue's instruction folders and return its zip episodes."""
    episodes: list[EpisodeRef] = []
    m = re.search(r"tissue_(\d+)", tissue_dir.name)
    tissue_index = int(m.group(1)) if m else None
    for instr_dir in sorted(p for p in tissue_dir.iterdir() if p.is_dir()):
        instruction, is_recovery = _base._clean_instruction_name(instr_dir.name)
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
    return episodes


def _extract_ts_id(zip_path: Path) -> Optional[str]:
    m = _TS_ID_RE.search(zip_path.name)
    return m.group(1) if m else None


def _read_ts_ids_from_markdown(md_path: Path) -> set[str]:
    """Extract the set of episode timestamp IDs referenced by a Markdown
    file (typically ``README_duplicates.md`` of the reference LeRobot
    dataset).  The regex matches every ``YYYYMMDD-HHMMSS-UUUUUU`` token in
    the file."""
    return set(_TS_ID_RE.findall(md_path.read_text()))


def discover_episodes_for_subset(
    subset: SubsetConfig,
    input_base: Path,
    *,
    whitelist_path_override: Optional[Path] = None,
) -> list[EpisodeRef]:
    """Discover the episodes that belong to one ``SubsetConfig``.

    Walks every source subpath (handling both tissue-dir and
    parent-of-tissues layouts), concatenates the resulting episode list,
    sorts by zip path for deterministic ordering, and finally applies the
    timestamp-ID whitelist (if any).

    Parameters
    ----------
    subset :
        The subset to discover.
    input_base :
        Root path under which ``subset.source_subpaths`` resolve.
    whitelist_path_override :
        If given, overrides ``subset.whitelist_md_path``.  Use this to
        point at a different ``README_duplicates.md`` (e.g. when running
        from a different cluster mount layout).
    """
    all_eps: list[EpisodeRef] = []
    for src_rel in subset.source_subpaths:
        src = (input_base / src_rel.lstrip("/")).resolve()
        tissue_dirs = _walk_tissue_dirs(src)
        if not tissue_dirs:
            print(f"[{subset.name}] WARN: no tissue_* dirs under {src}")
            continue
        for td in tissue_dirs:
            all_eps.extend(_collect_tissue_episodes(td))

    # Sort for deterministic merge order in the shard-and-merge pipeline.
    all_eps.sort(key=lambda ep: str(ep.zip_path))

    # Resolve effective whitelist path: explicit override > subset config.
    wl_path: Optional[Path] = None
    if whitelist_path_override is not None:
        wl_path = whitelist_path_override
    elif subset.whitelist_md_path is not None:
        wl_path = Path(subset.whitelist_md_path)

    if wl_path is None:
        return all_eps

    if not wl_path.exists():
        print(
            f"[{subset.name}] WARN: whitelist file not found at {wl_path}.\n"
            f"[{subset.name}] WARN: falling back to NO whitelist filter — "
            f"this will produce a SUPERSET of the reference dataset.  Pass "
            f"--hf-suturebot-readme to point at a real README_duplicates.md."
        )
        return all_eps

    ts_ids = _read_ts_ids_from_markdown(wl_path)
    print(
        f"[{subset.name}] loaded {len(ts_ids)} timestamp IDs from "
        f"{wl_path}"
    )

    before = len(all_eps)
    kept = [e for e in all_eps if (_extract_ts_id(e.zip_path) or "") in ts_ids]
    dropped = before - len(kept)
    print(
        f"[{subset.name}] whitelist filter: {before} → {len(kept)} episodes "
        f"({dropped} dropped, expected scope: 1452)"
    )
    return kept


# ---------------------------------------------------------------------------
# Per-subset and CLI
# ---------------------------------------------------------------------------


def convert_one_subset(
    subset: SubsetConfig,
    *,
    input_base: Path,
    output_root: Optional[Path],
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
    batch_encoding_size: int,
    max_episodes: Optional[int],
    num_workers: int,
    whitelist_path_override: Optional[Path],
) -> Optional[Path]:
    """Discover + convert one subset.  Returns the output dataset path
    (``None`` if there were no episodes to convert).
    """
    episodes = discover_episodes_for_subset(
        subset,
        input_base=input_base,
        whitelist_path_override=whitelist_path_override,
    )
    if not episodes:
        print(f"[{subset.name}] no episodes to convert; skipping")
        return None

    # ``source_path_str`` shows up in the README written into the dataset.
    # For multi-root subsets we list every input root, joined by " + ".
    source_path_str = " + ".join(
        str((input_base / sp.lstrip("/")).resolve())
        for sp in subset.source_subpaths
    )

    return convert_subset(
        episodes=episodes,
        dataset_name=subset.name,
        output_root=output_root,
        overwrite=overwrite,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
        max_episodes=max_episodes,
        source_path_str=source_path_str,
        num_workers=num_workers,
    )


def main(
    input_base: Path = Path(JHU_PREFIX_DEFAULT),
    output_root: Optional[Path] = None,
    subsets: list[str] = list(s.name for s in SUBSETS_TO_CONVERT_HIGHRES),
    single_zip: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    task: str = "dvrk suturing task",
    max_episodes: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_writer_processes: int = 8,
    image_writer_threads: int = 16,
    batch_encoding_size: int = 1,
    overwrite: bool = True,
    hf_suturebot_readme: Optional[Path] = None,
) -> None:
    """Convert JHU dVRK zarr subsets to LeRobot v2.1 at high resolution
    (720 H x 960 W).

    Args:
        input_base: Root of the JHU zarr subsets.  Source subpaths are
            resolved relative to this.
        output_root: Where to write the converted LeRobot datasets.
            Defaults to ``$HF_LEROBOT_HOME``.  One subdirectory is
            created per subset.
        subsets: Which output subsets to produce, by name.  Default: all
            9 entries in ``SUBSETS_TO_CONVERT_HIGHRES``.  Pass any subset
            of ``[knot_tying, suture_bot_success, suture_bot_failure,
            ood, cosmos_fail_filtered, cosmos_throw_fail_demo,
            cosmos_knot_fail_demo, suturebot_act_throw_eval,
            hf_suturebot]``.
        single_zip: Debug mode — convert a single ``.zip`` directly into a
            standalone dataset, bypassing all subset discovery and
            whitelist logic.  Mutually exclusive with ``--subsets``.
        dataset_name: With ``--single-zip``, the output dataset name
            (defaults to the zip's stem).  Ignored otherwise.
        task: With ``--single-zip``, the instruction string.
        max_episodes: Limit the number of episodes converted per subset
            (debug).
        num_workers: Number of parallel shards for shard-and-merge
            conversion.  Defaults to ``min(8, cpu_count() - 1)``.  Pass
            ``1`` for sequential (no shard merge).
        image_writer_processes: LeRobot internal video-encoder pool size.
        image_writer_threads: Threads per video encoder.
        batch_encoding_size: Episodes per video-encode batch (1 is safest;
            see the lower-res script's docstring for caveats).
        overwrite: If True, remove any existing output dataset directory
            before writing.
        hf_suturebot_readme: Override the default cluster path of
            ``hf_suturebot/README_duplicates.md`` (the source of truth
            for the 1452-episode whitelist).  Applies only to the
            ``hf_suturebot`` subset.
    """
    if num_workers is None:
        num_workers = default_num_workers()
    if num_workers < 1:
        num_workers = 1

    print(
        f"[highres] target resolution: {TARGET_HEIGHT} (H) x "
        f"{TARGET_WIDTH} (W)  (16:9 source 540x960 letterboxed top+bottom)"
    )

    if single_zip is not None:
        # Bypass subset machinery entirely — useful for spot-checking the
        # high-res output on a single zarr zip.
        if not single_zip.exists():
            print(f"ERROR: --single-zip does not exist: {single_zip}")
            sys.exit(1)
        episode = EpisodeRef(
            zip_path=single_zip,
            instruction=task,
            tissue_index=None,
            subtask_name="single_zip",
            is_recovery=False,
        )
        resolved_name = dataset_name or f"{single_zip.stem}_highres"
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
        return

    if not input_base.exists():
        print(f"ERROR: --input-base does not exist: {input_base}")
        sys.exit(1)

    # Resolve subset names.
    try:
        chosen_subsets = [_subset_by_name(n) for n in subsets]
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    wl_override = Path(hf_suturebot_readme) if hf_suturebot_readme else None

    print(
        f"[highres] processing {len(chosen_subsets)} subset(s): "
        f"{[s.name for s in chosen_subsets]}"
    )

    results: list[tuple[str, Optional[Path], int]] = []
    t0 = time.time()
    for subset in chosen_subsets:
        # Whitelist override only meaningful for hf_suturebot; harmless
        # to forward to other subsets (which have whitelist_md_path=None
        # so the override is ignored unless explicitly used).
        per_subset_override = (
            wl_override if subset.name == "hf_suturebot" else None
        )
        try:
            output_path = convert_one_subset(
                subset,
                input_base=input_base,
                output_root=output_root,
                overwrite=overwrite,
                image_writer_processes=image_writer_processes,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=batch_encoding_size,
                max_episodes=max_episodes,
                num_workers=num_workers,
                whitelist_path_override=per_subset_override,
            )
            results.append((subset.name, output_path, 0))
        except Exception as e:  # noqa: BLE001
            print(f"[{subset.name}] ERROR: {e}")
            traceback.print_exc()
            results.append((subset.name, None, 1))

    elapsed = time.time() - t0
    print(f"\n=== high-res conversion summary (in {elapsed:.1f}s) ===")
    if not results:
        print("No subsets were converted.")
        return
    for name, path, err in results:
        if err:
            print(f"  {name}: FAILED")
        elif path is None:
            print(f"  {name}: skipped (no episodes)")
        else:
            print(f"  {name}: → {path}")


if __name__ == "__main__":
    tyro.cli(main)
