#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Keyboard-driven OOD scenario generator for the Cosmos-H-Surgical-Simulator.

This is a checkpoint quality gate: it runs **offline** (not real-time) against
any C-H-S-S checkpoint and renders short videos that simulate what the model
would produce if a user held particular keys on the keyboard for ~5 seconds
during real-time deployment in ``flashsim-jg``.  It is meant as an *early
indicator* of how a teacher (or distilled student) checkpoint will behave when
driven by ``scripts/keyboard/keyboard_controller.py`` in the live simulator,
without needing to wire up the live pipeline.

Relationship to existing scripts:

* ``scripts/generate_open_h_ood_scenarios_depth.py`` -- this script reuses its
  trajectory rollout and reference-video machinery (``rollout_trajectory``,
  ``compute_action_statistics``, ``_save_short_reference_videos``, the
  ``ARM_LAYOUTS`` table) so the offline plumbing is shared.
* ``scripts/keyboard/keyboard_controller.py`` -- this script reuses its
  semantic-key -> action-dim ``_arm_binds(...)`` table and ``ControllerGains``
  defaults so the offline trajectories are byte-identical to what the live
  controller will emit when a user holds the same keys in the simulator.

What is *new* here, relative to the existing OOD script, is testing channels
that the existing script never touches: rotation (Q/E/T/G/C/V/U/O/J/L/N/M),
gripper (Space, Z, R-Shift, R-Ctrl), modifier scaling (L-Shift, L-Ctrl), and
multi-key compositional combos (e.g. up + push depth, up + close gripper).

Anchor convention: each step of every scenario is built as
``mean_action + sum-of-key-offsets`` where ``mean_action`` is the empirical
per-dataset mean of transformed actions sampled from the test split.  The data
is mean-std-normalised so ``mean_action`` is numerically close to zero, but
using the actual empirical mean (i) removes any small dataset bias, and (ii)
keeps every step inside the action distribution the model was trained on.  This
matches the convention of the existing OOD script for direct comparability;
the live keyboard controller currently anchors at the zero vector (which is
near-identical because the action space is normalised).

Output structure (one timestamped run per invocation)::

    <save_root>/<TIMESTAMP>/
    +-- README.md                                  <-- legend for every scenario
    +-- run_meta.json                              <-- full reproducibility record
    +-- <dataset_name>/
        +-- gt/episode_XXXX.mp4                    (6 chunks = 72 frames)
        +-- gt_60pred/episode_XXXX.mp4             (matches scenario length)
        +-- predicted/episode_XXXX.mp4             (autoregressive w/ GT actions)
        +-- predicted_60pred/episode_XXXX.mp4
        +-- keyboard_scenarios/episode_XXXX/
            +-- 01_R_up.mp4
            +-- 02_R_down.mp4
            +-- ...
            +-- 20_both_walk_grip.mp4
            +-- scenarios.json

Standard usage (one invocation, all 20 scenarios per episode)::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \\
        scripts/generate_keyboard_input_scenarios.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss \\
        --ckpt_path /lustre/.../checkpoints/iter_000004000/model_ema_bf16.pt \\
        --test_episodes_json output/jhu_dvrk_mono_test_episodes.json \\
        --episodes_per_dataset 1 \\
        --save_root results/keyboard_input_scenarios

CLI knobs of note:

* ``--scenarios <NAME...>``: run only the listed scenarios (e.g. ``01_R_up
  09_R_push 13_R_grip_close``) for fast iteration on a specific control axis.
* ``--gains_*``: override any single ``ControllerGains`` field (sigma units).
* ``--anchor_at_zero``: switch from empirical-mean anchoring to zero-vector
  anchoring (matches the live ``KeyboardController`` exactly).  Defaults off.

Estimated cost: ~5-7 min/episode for the 16-scenario default on a single H100;
plus ~1-2 min/episode for the GT + predicted reference videos, so ~10 min per
(dataset, episode) pair total.  3 datasets x 1 ep ~= 30 min single-GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    MAX_ACTION_DIM,
)
from cosmos_predict2._src.predict2.action.inference.inference_open_h import (
    CHUNK_SIZE,
    build_episode_index_map,
    find_chunk_indices,
    pad_action,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)

from scripts.cosmos_h_surgical_simulator_quant_eval import (
    _lookup_exclude_splits,
    load_wrapped_dataset,
    resolve_timestep_interval,
)
from scripts.generate_open_h_ood_scenarios_depth import (
    ARM_LAYOUTS,
    NUM_GT_CHUNKS,
    NUM_OOD_CHUNKS,
    OOD_TRAJECTORY_LENGTH,
    _auto_detect_arm_layout,
    _save_short_reference_videos,
    compute_action_statistics,
    rollout_trajectory,
)
from scripts.keyboard.keyboard_controller import (
    DVRK_RAW_DIM,
    ControllerGains,
    _arm_binds,
    _Bind,
    _LEFT_BASE,
    _RIGHT_BASE,
)


# Reusable handles into the keyboard controller's bind table -- built once at
# module import.  Each value is a list of _Bind because semantic keys like
# "yaw_pos" perturb >1 rot6d dim simultaneously.
_LEFT_ARM_BINDS: dict[str, list[_Bind]] = _arm_binds(_LEFT_BASE)
_RIGHT_ARM_BINDS: dict[str, list[_Bind]] = _arm_binds(_RIGHT_BASE)


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyboardSegment:
    """One sub-segment of a scenario: a constant set of held keys for ``num_steps``.

    Held keys are SEMANTIC keys (the keys of ``_arm_binds(...)``), e.g.
    ``"y_pos"``, ``"z_neg"``, ``"yaw_pos"``, ``"grip_close"``.  Use the
    physical-key documentation in :class:`KeyboardScenario` (and the auto-
    generated README) for the human-readable mapping.

    Attributes:
        held_left:  Tuple of semantic keys driving the LEFT (PSM2) arm.
        held_right: Tuple of semantic keys driving the RIGHT (PSM1) arm.
        fast: If True, multiply translation+rotation channels by
            ``ControllerGains.fast_multiplier`` (mirrors L-Shift in the live
            controller).  Gripper is unscaled.  Mutually exclusive with
            ``slow``.
        slow: If True, multiply translation+rotation channels by
            ``ControllerGains.slow_multiplier`` (mirrors L-Ctrl).  Gripper is
            unscaled.  Mutually exclusive with ``fast``.
        num_steps: Number of identical action steps emitted for this segment.
            Within one :class:`KeyboardScenario`, all segment ``num_steps``
            must sum to ``OOD_TRAJECTORY_LENGTH`` (60).
    """

    held_left: tuple[str, ...] = ()
    held_right: tuple[str, ...] = ()
    fast: bool = False
    slow: bool = False
    num_steps: int = OOD_TRAJECTORY_LENGTH

    def __post_init__(self) -> None:
        if self.fast and self.slow:
            raise ValueError("KeyboardSegment cannot be both fast and slow")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")


@dataclass(frozen=True)
class KeyboardScenario:
    """A complete keyboard-driven scenario: one or more sequenced segments.

    The total step count across all segments must be exactly
    ``OOD_TRAJECTORY_LENGTH`` so the trajectory matches the OOD-script
    rollout length (5 chunks x 12 = 60 steps).
    """

    filename: str           # e.g. "01_R_up" -- becomes the .mp4 stem
    physical_keys: str      # e.g. "Up"; for the README and metadata only
    description: str        # human-readable, used in the README
    test: str               # 1-line summary of what behavior this probes
    segments: tuple[KeyboardSegment, ...]

    def __post_init__(self) -> None:
        total = sum(s.num_steps for s in self.segments)
        if total != OOD_TRAJECTORY_LENGTH:
            raise ValueError(
                f"Scenario {self.filename}: segment steps sum to {total}, "
                f"expected {OOD_TRAJECTORY_LENGTH}"
            )


# ---------------------------------------------------------------------------
# The 20 scenarios (dual-arm-rich; single-arm embodiments are filtered down
# automatically -- see ``_filter_scenarios_for_arm_layout``).
#
# Numbering is sequential 01-20 and grouped by function:
#   01-04 : single-arm RIGHT translation       (y+, y-, x-, x+)
#   05-08 : DUAL translation (y-axis)          (both up, R-up_L-down, L-up_R-down, both down)
#   09-10 : single-arm RIGHT depth             (push, pull)
#   11-13 : DUAL depth                         (both push, both pull, L-push_R-pull)
#   14-15 : single-arm RIGHT rotation          (yaw+, pitch+)
#   16-17 : single-arm + DUAL gripper          (R close, both close)
#   18    : sequenced reversal                 (R up -> R down)
#   19-20 : compound motion + gripper          (R walk-and-grip, dual walk-and-grip)
#
# Earlier drafts had isolated LEFT-arm-only sanity checks (#05_L_up, #06_L_down)
# and a modifier-scaling test (#15_R_up_fast).  They were dropped after the
# dual-arm coverage expansion because (a) every new dual-arm scenario already
# moves the LEFT arm so the isolated tests became redundant, and (b) modifier
# scaling is a pure controller feature, not a model capability per se -- if
# 01_R_up works at 1 sigma the model will not fail uniquely at 2 sigma.
# ---------------------------------------------------------------------------

DUAL_ARM_SCENARIOS: tuple[KeyboardScenario, ...] = (
    # ----- 01-04: single-arm RIGHT translation (arrow keys, dims 0-9) ----
    KeyboardScenario(
        filename="01_R_up",
        physical_keys="Up arrow",
        description="Hold Up arrow -> R-arm moves up (image y+) for 60 steps.",
        test="single-axis translation, RIGHT arm, image-y positive",
        segments=(KeyboardSegment(held_right=("y_pos",)),),
    ),
    KeyboardScenario(
        filename="02_R_down",
        physical_keys="Down arrow",
        description="Hold Down arrow -> R-arm moves down (image y-) for 60 steps.",
        test="single-axis translation, RIGHT arm, image-y negative",
        segments=(KeyboardSegment(held_right=("y_neg",)),),
    ),
    KeyboardScenario(
        filename="03_R_left",
        physical_keys="Left arrow",
        description="Hold Left arrow -> R-arm moves left (image x-) for 60 steps.",
        test="single-axis translation, RIGHT arm, image-x negative",
        segments=(KeyboardSegment(held_right=("x_neg",)),),
    ),
    KeyboardScenario(
        filename="04_R_right",
        physical_keys="Right arrow",
        description="Hold Right arrow -> R-arm moves right (image x+) for 60 steps.",
        test="single-axis translation, RIGHT arm, image-x positive",
        segments=(KeyboardSegment(held_right=("x_pos",)),),
    ),
    # ----- 05-08: DUAL translation, y-axis (the 2x2 of {L+R} x {up,down}) -
    KeyboardScenario(
        filename="05_both_up",
        physical_keys="Up + W",
        description="Hold Up + W -> BOTH arms move up simultaneously for 60 steps.",
        test="dual-arm coordinated up; closest analogue to OOD #03",
        segments=(KeyboardSegment(held_left=("y_pos",), held_right=("y_pos",)),),
    ),
    KeyboardScenario(
        filename="06_R_up_L_down",
        physical_keys="Up + S",
        description="Hold Up + S -> R-arm up while L-arm down for 60 steps.",
        test="dual-arm OPPOSING motion; probes per-arm conditioning",
        segments=(KeyboardSegment(held_left=("y_neg",), held_right=("y_pos",)),),
    ),
    KeyboardScenario(
        filename="07_L_up_R_down",
        physical_keys="W + Down arrow",
        description="Hold W + Down -> L-arm up while R-arm down for 60 steps.",
        test="dual-arm OPPOSING motion (mirror of #06); probes per-arm symmetry",
        segments=(KeyboardSegment(held_left=("y_pos",), held_right=("y_neg",)),),
    ),
    KeyboardScenario(
        filename="08_both_down",
        physical_keys="S + Down arrow",
        description="Hold S + Down -> BOTH arms move down simultaneously for 60 steps.",
        test="dual-arm coordinated down; closest analogue to OOD #04",
        segments=(KeyboardSegment(held_left=("y_neg",), held_right=("y_neg",)),),
    ),
    # ----- 09-10: single-arm RIGHT depth (live controller uses I/K) -----
    KeyboardScenario(
        filename="09_R_push",
        physical_keys="I",
        description="Hold I -> R-arm pushes into tissue (z+) for 60 steps.",
        test="single-arm depth into tissue; live-controller analogue of OOD #13",
        segments=(KeyboardSegment(held_right=("z_pos",)),),
    ),
    KeyboardScenario(
        filename="10_R_pull",
        physical_keys="K",
        description="Hold K -> R-arm pulls away from tissue (z-) for 60 steps.",
        test="single-arm depth away; live-controller analogue of OOD #14",
        segments=(KeyboardSegment(held_right=("z_neg",)),),
    ),
    # ----- 11-13: DUAL depth (push/pull/anti-phase) ---------------------
    KeyboardScenario(
        filename="11_both_push",
        physical_keys="R + I",
        description=(
            "Hold R + I -> BOTH arms push into tissue (z+) simultaneously for 60 steps."
        ),
        test=(
            "dual-arm coordinated depth-in; analogue of OOD #13 via the "
            "live-controller key combination"
        ),
        segments=(KeyboardSegment(held_left=("z_pos",), held_right=("z_pos",)),),
    ),
    KeyboardScenario(
        filename="12_both_pull",
        physical_keys="F + K",
        description=(
            "Hold F + K -> BOTH arms pull away from tissue (z-) simultaneously for 60 steps."
        ),
        test=(
            "dual-arm coordinated depth-out; analogue of OOD #14 via the "
            "live-controller key combination"
        ),
        segments=(KeyboardSegment(held_left=("z_neg",), held_right=("z_neg",)),),
    ),
    KeyboardScenario(
        filename="13_L_push_R_pull",
        physical_keys="R + K",
        description=(
            "Hold R + K -> L-arm pushes (+z) while R-arm pulls (-z) for 60 steps."
        ),
        test=(
            "dual-arm ANTI-PHASE depth; strongest probe of per-arm depth "
            "conditioning -- analogue of OOD #15"
        ),
        segments=(KeyboardSegment(held_left=("z_pos",), held_right=("z_neg",)),),
    ),
    # ----- 14-15: single-arm RIGHT rotation (no analogue in the OOD script)
    KeyboardScenario(
        filename="14_R_yaw_pos",
        physical_keys="O",
        description="Hold O -> R-arm yaws +Z for 60 steps (rot6d dims 1, 3).",
        test="ROTATION (yaw); fully OOD vs the existing OOD scenario suite",
        segments=(KeyboardSegment(held_right=("yaw_pos",)),),
    ),
    KeyboardScenario(
        filename="15_R_pitch_pos",
        physical_keys="J",
        description="Hold J -> R-arm pitches +Y for 60 steps (rot6d dim 2).",
        test="ROTATION (pitch); fully OOD vs the existing OOD scenario suite",
        segments=(KeyboardSegment(held_right=("pitch_pos",)),),
    ),
    # ----- 16-17: gripper (single + dual; no analogue in the OOD script) --
    KeyboardScenario(
        filename="16_R_grip_close",
        physical_keys="Right Shift",
        description="Hold R-Shift -> R-arm gripper closes (held) for 60 steps.",
        test="GRIPPER close, single arm; fully OOD vs the existing OOD scenario suite",
        segments=(KeyboardSegment(held_right=("grip_close",)),),
    ),
    KeyboardScenario(
        filename="17_both_grip_close",
        physical_keys="Space + Right Shift",
        description=(
            "Hold Space + R-Shift -> BOTH arms close gripper simultaneously for 60 steps."
        ),
        test=(
            "GRIPPER close, dual arm (suturebot-style needle grasp); fully OOD "
            "vs every existing scenario suite"
        ),
        segments=(KeyboardSegment(held_left=("grip_close",), held_right=("grip_close",)),),
    ),
    # ----- 18: sequenced reversal (R-arm up then down) ------------------
    KeyboardScenario(
        filename="18_R_up_then_down",
        physical_keys="Up (30) -> Down (30)",
        description="R-arm up for 30 steps, then R-arm down for 30 steps.",
        test="REVERSAL recovery; checks if model handles an abrupt sign flip",
        segments=(
            KeyboardSegment(held_right=("y_pos",), num_steps=30),
            KeyboardSegment(held_right=("y_neg",), num_steps=30),
        ),
    ),
    # ----- 19-20: compound motion + gripper (single + dual walk-and-grip)
    KeyboardScenario(
        filename="19_R_walk_grip",
        physical_keys="Up + Right Shift",
        description="Hold Up + R-Shift -> R-arm walks up with gripper closed for 60 steps.",
        test="REALISTIC compound: simultaneous R-arm motion + R-arm gripper",
        segments=(KeyboardSegment(held_right=("y_pos", "grip_close")),),
    ),
    KeyboardScenario(
        filename="20_both_walk_grip",
        physical_keys="W + Up + Space + Right Shift",
        description=(
            "Hold W + Up + Space + R-Shift -> BOTH arms walk up with BOTH "
            "grippers closed for 60 steps."
        ),
        test=(
            "REALISTIC compound at full dual-arm scope (suturing posture: "
            "both arms moving while both grippers grasp)"
        ),
        segments=(
            KeyboardSegment(
                held_left=("y_pos", "grip_close"),
                held_right=("y_pos", "grip_close"),
            ),
        ),
    ),
)


# Scenarios that involve ONLY the right arm survive the dual-arm filter.
# (The keyboard controller is dual-arm-only by design; on a single-arm
# embodiment we degrade gracefully by silently zero-ing left-arm offsets.)
def _filter_scenarios_for_arm_layout(
    scenarios: tuple[KeyboardScenario, ...],
    arm_layout: dict,
) -> tuple[KeyboardScenario, ...]:
    """If the embodiment is single-arm, drop scenarios that need both arms.

    The action-dim layout for single-arm embodiments puts the only arm at the
    LEFT slot (10-19) per :data:`ARM_LAYOUTS`, so right-arm-only scenarios
    would render no motion at all -- skip them rather than mislead the user.
    """
    if arm_layout.get("dual", False):
        return scenarios
    keep: list[KeyboardScenario] = []
    for s in scenarios:
        uses_left = any(seg.held_left for seg in s.segments)
        uses_right = any(seg.held_right for seg in s.segments)
        if uses_right and not uses_left:
            continue
        keep.append(s)
    return tuple(keep)


# ---------------------------------------------------------------------------
# Trajectory builder (offline counterpart of KeyboardController._build_action_step)
# ---------------------------------------------------------------------------

def _gain_for_bind(bind: _Bind, gains: ControllerGains, modifier: float) -> float:
    """Return the signed sigma offset for a single :class:`_Bind`.

    Mirrors the per-kind logic in
    ``KeyboardController._build_action_step``: translation channels are
    scaled by ``translation_xy`` or ``translation_z`` (z is the third dim
    within each arm's 10-D block); rotation channels by ``rotation``;
    gripper channels by ``gripper`` and are NOT affected by the speed
    modifier (the live controller deliberately decouples gripper response
    from the fast/slow modifiers).
    """
    if bind.kind == "translation":
        # z occupies dim%10 == 2 within each arm's 10-D block (dims 2 and 12).
        is_z = (bind.dim % 10 == 2)
        gain = gains.translation_z if is_z else gains.translation_xy
        return bind.coef * gain * modifier
    if bind.kind == "rotation":
        return bind.coef * gains.rotation * modifier
    if bind.kind == "gripper":
        return bind.coef * gains.gripper
    raise ValueError(f"Unknown bind.kind={bind.kind!r}")


def build_keyboard_trajectory(
    scenario: KeyboardScenario,
    gains: ControllerGains,
    mean_action: np.ndarray,
    raw_dim: int,
    anchor_at_zero: bool = False,
) -> np.ndarray:
    """Build a ``(OOD_TRAJECTORY_LENGTH, raw_dim)`` action trajectory.

    The trajectory is the concatenation of per-segment tiles, where each
    segment's action step is::

        action = anchor[0:raw_dim] + sum_over_held_keys(bind.coef * gain * modifier)

    ``anchor`` is either the empirical per-dataset mean (default) or the
    zero vector (with ``anchor_at_zero=True``, matching the live keyboard
    controller).

    Args:
        scenario: KeyboardScenario describing the held-key sequence.
        gains: Per-channel gain table (units of training-sigma).
        mean_action: Empirical action mean, shape ``(raw_dim,)``.  Used as
            the per-step anchor when ``anchor_at_zero=False``.
        raw_dim: Un-padded action dimensionality.  Must be at least
            :data:`DVRK_RAW_DIM`; scenarios only write into the first
            ``DVRK_RAW_DIM`` channels (the rest stay at the anchor value).
        anchor_at_zero: If True, replace ``mean_action`` with a zero vector
            of the same shape -- matches the current live ``KeyboardController``
            exactly but loses any small dataset-mean bias correction.

    Returns:
        ``np.ndarray`` of shape ``(OOD_TRAJECTORY_LENGTH, raw_dim)`` and
        dtype ``float32``.

    Raises:
        ValueError: if ``raw_dim < DVRK_RAW_DIM`` (the underlying bind table
            assumes 20-D dVRK channels exist).
    """
    if raw_dim < DVRK_RAW_DIM:
        raise ValueError(
            f"raw_dim={raw_dim} too small for dVRK keyboard scenarios "
            f"(need >= {DVRK_RAW_DIM})"
        )
    if mean_action.shape[0] < raw_dim:
        raise ValueError(
            f"mean_action shape {mean_action.shape} smaller than raw_dim={raw_dim}"
        )

    anchor = (
        np.zeros_like(mean_action[:raw_dim], dtype=np.float32)
        if anchor_at_zero
        else mean_action[:raw_dim].astype(np.float32, copy=True)
    )

    chunks: list[np.ndarray] = []
    for segment in scenario.segments:
        if segment.fast:
            modifier = gains.fast_multiplier
        elif segment.slow:
            modifier = gains.slow_multiplier
        else:
            modifier = 1.0

        step = anchor.copy()

        for sem in segment.held_left:
            if sem not in _LEFT_ARM_BINDS:
                raise ValueError(
                    f"Scenario {scenario.filename}: unknown LEFT semantic key "
                    f"'{sem}' (valid: {sorted(_LEFT_ARM_BINDS)})"
                )
            for bind in _LEFT_ARM_BINDS[sem]:
                if bind.dim < raw_dim:  # silently skip dims the embodiment lacks
                    step[bind.dim] += _gain_for_bind(bind, gains, modifier)

        for sem in segment.held_right:
            if sem not in _RIGHT_ARM_BINDS:
                raise ValueError(
                    f"Scenario {scenario.filename}: unknown RIGHT semantic key "
                    f"'{sem}' (valid: {sorted(_RIGHT_ARM_BINDS)})"
                )
            for bind in _RIGHT_ARM_BINDS[sem]:
                if bind.dim < raw_dim:
                    step[bind.dim] += _gain_for_bind(bind, gains, modifier)

        chunks.append(np.tile(step, (segment.num_steps, 1)))

    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# README writer (auto-generates the legend at run-time so users get it next to the videos)
# ---------------------------------------------------------------------------

def _format_gains_block(gains: ControllerGains) -> list[str]:
    return [
        f"  translation_xy = {gains.translation_xy} sigma",
        f"  translation_z  = {gains.translation_z} sigma  (depth; data std is small)",
        f"  rotation       = {gains.rotation} sigma  (per-axis L2 norm)",
        f"  gripper        = {gains.gripper} sigma  (absolute target while held)",
        f"  fast_modifier  = {gains.fast_multiplier}x  (Left-Shift)",
        f"  slow_modifier  = {gains.slow_multiplier}x  (Left-Ctrl)",
    ]


def _write_readme(
    save_root: str,
    scenarios: Sequence[KeyboardScenario],
    gains: ControllerGains,
    anchor_at_zero: bool,
    ckpt_path: str,
    experiment: str,
) -> str:
    """Auto-generate a Markdown README explaining the scenario layout.

    Returns the absolute path of the written README.
    """
    anchor_label = "zero vector (matches live KeyboardController exactly)" if anchor_at_zero \
        else "empirical per-dataset mean (matches existing OOD-script convention)"

    lines: list[str] = [
        "# Keyboard-input OOD scenarios -- output legend",
        "",
        f"_Run timestamp: {os.path.basename(save_root)}_",
        "",
        f"_Checkpoint: `{ckpt_path}`_",
        "",
        f"_Experiment: `{experiment}`_",
        "",
        "## Purpose",
        "",
        "These videos are an offline checkpoint quality gate that simulates what",
        "the model would render if a user held particular keys on the keyboard",
        "for ~5 seconds during real-time deployment in `flashsim-jg`.  Use them",
        "to spot-check whether the candidate teacher (or distilled student)",
        "checkpoint will follow keyboard-issued trajectories before wiring up",
        "the live pipeline.  This is NOT a real-time test.",
        "",
        "## How the trajectories are built",
        "",
        "Each scenario emits a 60-step action trajectory (5 chunks of 12 = ~5 s",
        "at 12 fps).  Per step, the action vector is::",
        "",
        f"    action = {('zeros' if anchor_at_zero else 'mean_action')} + sum(bind.coef * gain * modifier)",
        "",
        f"...where the anchor is the **{anchor_label}**, and each held",
        "semantic key contributes a sigma-unit offset taken from",
        "`scripts/keyboard/keyboard_controller.py::_arm_binds(...)`.  The",
        "mapping from physical keys (e.g. `Up arrow`) to semantic keys (e.g.",
        "`y_pos` for the right arm) is the live controller's own mapping; see",
        "the controller's docstring for the full key map.",
        "",
        "Default gains (`ControllerGains`):",
        "",
        *_format_gains_block(gains),
        "",
        "Modifier semantics: `fast` and `slow` scale translation+rotation",
        "channels but not gripper, mirroring the live controller (held-key",
        "gripper response is deliberately decoupled from speed).",
        "",
        "## Scenarios",
        "",
        "| #  | File | Physical keys | Description | Tests |",
        "|----|------|---------------|-------------|-------|",
    ]
    for s in scenarios:
        lines.append(
            f"| {s.filename[:2]} | `{s.filename}.mp4` | "
            f"`{s.physical_keys}` | {s.description} | _{s.test}_ |"
        )

    lines.extend([
        "",
        "## Layout per dataset",
        "",
        "```",
        "<save_root>/<TIMESTAMP>/<dataset_name>/",
        "+-- gt/episode_XXXX.mp4              # ground truth, 6 chunks (72 frames)",
        "+-- gt_60pred/episode_XXXX.mp4       # gt clipped to 60 frames + 1 cond",
        "+-- predicted/episode_XXXX.mp4       # autoregressive prediction with GT actions",
        "+-- predicted_60pred/episode_XXXX.mp4",
        "+-- keyboard_scenarios/episode_XXXX/",
        "    +-- 01_R_up.mp4",
        "    +-- 02_R_down.mp4",
        "    +-- ...",
        "    +-- 20_both_walk_grip.mp4",
        "    +-- scenarios.json               # per-scenario metadata",
        "```",
        "",
        "## Reading the videos",
        "",
        "Open each scenario `.mp4` next to `predicted_60pred/episode_XXXX.mp4`",
        "(the model's output for the same starting frame with the recorded",
        "actions) and `gt_60pred/episode_XXXX.mp4` (what really happens).  For",
        "each scenario, look for:",
        "",
        "* **Smooth, deterministic motion in the held direction** -- good.",
        "* **Tool deformation, colour drift, geometry tearing** -- bad.",
        "* **Static / frozen output despite a held key** -- the model is",
        "  collapsing on this OOD command; bad sign for live deployment.",
        "* **The non-commanded arm staying still** when only one arm is",
        "  driven (e.g. #01-#04, #09-#13) -- good; confirms per-arm",
        "  conditioning has been learnt.",
        "* **Reversal latency** in #14 -- a few frames of overshoot are normal,",
        "  but >12 frames (one chunk) of \"still moving up\" after the key flips",
        "  to Down is a red flag.",
        "* **Modifier amplification** in #15: the same R-up command should",
        "  produce visibly more motion than #01.",
        "* **Compositional integrity** in #07/#08/#16: simultaneous commands",
        "  should produce simultaneous responses, not a compromise.",
        "",
        "## Scenarios.json fields",
        "",
        "Each `keyboard_scenarios/episode_XXXX/scenarios.json` records, for",
        "every scenario rolled out: `filename`, `description`, `physical_keys`,",
        "the `held_left` / `held_right` / `fast` / `slow` / `num_steps` of",
        "every segment, and the inference `seed` / wall-time used.  This is",
        "the source of truth for downstream comparison scripts.",
        "",
    ])

    readme_path = os.path.join(save_root, "README.md")
    os.makedirs(save_root, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"README written to {readme_path}")
    return readme_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Mirrors ``generate_open_h_ood_scenarios_depth.py`` for consistency, with
    extra flags for the keyboard-specific gains and scenario selection.  No
    default for ``--ckpt_path`` -- callers must specify which checkpoint to
    test (per design).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate keyboard-input OOD scenario videos as a pre-deployment "
            "checkpoint quality gate for the C-H-S-S action-conditioned model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model ---
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to checkpoint (.pt file).  No default -- specify per-run.")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")

    # --- Dataset ---
    parser.add_argument("--test_episodes_json", type=str, required=True,
                        help="Path to test_episodes.json")
    parser.add_argument("--episodes_per_dataset", type=int, default=1,
                        help="Episodes per dataset (default: 1)")
    parser.add_argument("--exclude_datasets", type=str, nargs="+", default=None,
                        help="Dataset names to skip")

    # --- Inference ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance", type=float, default=0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Context parallel size (GPUs)")

    # --- Keyboard scenario selection ---
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=None,
        help=(
            "Run only the listed scenario filenames (without .mp4), e.g. "
            "'01_R_up 09_R_push 14_R_yaw_pos 16_R_grip_close 18_R_up_then_down 20_both_walk_grip'.  "
            "Default: run all 20."
        ),
    )

    # --- Anchor / gain overrides ---
    parser.add_argument(
        "--anchor_at_zero", action="store_true",
        help=(
            "Anchor every step at the zero vector (matches live "
            "KeyboardController exactly).  Default: anchor at empirical "
            "per-dataset mean (matches existing OOD-script convention)."
        ),
    )
    parser.add_argument("--gains_translation_xy", type=float, default=None,
                        help="Override translation_xy gain (sigma units)")
    parser.add_argument("--gains_translation_z", type=float, default=None,
                        help="Override translation_z gain (sigma units)")
    parser.add_argument("--gains_rotation", type=float, default=None,
                        help="Override rotation gain (sigma units)")
    parser.add_argument("--gains_gripper", type=float, default=None,
                        help="Override gripper gain (sigma units)")
    parser.add_argument("--gains_fast_multiplier", type=float, default=None,
                        help="Override fast (L-Shift) speed multiplier")
    parser.add_argument("--gains_slow_multiplier", type=float, default=None,
                        help="Override slow (L-Ctrl) speed multiplier")

    parser.add_argument("--stats_episodes", type=int, default=20,
                        help="Episodes to sample for action statistics (default: 20)")

    # --- Output ---
    parser.add_argument("--save_root", type=str, default="results/keyboard_input_scenarios",
                        help="Output directory")
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")

    return parser.parse_args()


def _build_gains_from_args(args: argparse.Namespace) -> ControllerGains:
    """Build a :class:`ControllerGains` honoring any explicit CLI overrides."""
    base = ControllerGains()
    return ControllerGains(
        translation_xy=args.gains_translation_xy if args.gains_translation_xy is not None else base.translation_xy,
        translation_z=args.gains_translation_z if args.gains_translation_z is not None else base.translation_z,
        rotation=args.gains_rotation if args.gains_rotation is not None else base.rotation,
        gripper=args.gains_gripper if args.gains_gripper is not None else base.gripper,
        fast_multiplier=args.gains_fast_multiplier if args.gains_fast_multiplier is not None else base.fast_multiplier,
        slow_multiplier=args.gains_slow_multiplier if args.gains_slow_multiplier is not None else base.slow_multiplier,
    )


def _select_scenarios(
    requested: Optional[Sequence[str]],
) -> tuple[KeyboardScenario, ...]:
    """Select scenarios from :data:`DUAL_ARM_SCENARIOS` by ``filename``.

    Raises ``ValueError`` for any unknown name so the user catches typos
    before the long inference loop starts.
    """
    if not requested:
        return DUAL_ARM_SCENARIOS

    by_name = {s.filename: s for s in DUAL_ARM_SCENARIOS}
    selected: list[KeyboardScenario] = []
    for name in requested:
        clean = name.removesuffix(".mp4")
        if clean not in by_name:
            valid = ", ".join(by_name)
            raise ValueError(f"Unknown scenario '{name}'.  Valid: {valid}")
        selected.append(by_name[clean])
    return tuple(selected)


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------

def run_episode_kbd(
    video2world: ActionVideo2WorldInference,
    dataset,
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    timestep_interval: int,
    arm_layout: dict,
    mean_action: np.ndarray,
    std_action: np.ndarray,
    raw_dim: int,
    save_dir: str,
    seed: int,
    guidance: float,
    save_fps: int,
    gains: ControllerGains,
    anchor_at_zero: bool,
    scenarios: Sequence[KeyboardScenario],
) -> int:
    """Process one episode: GT + predicted + N keyboard-driven scenarios.

    Output structure under *save_dir*:
        gt/episode_XXXX.mp4
        gt_60pred/episode_XXXX.mp4
        predicted/episode_XXXX.mp4
        predicted_60pred/episode_XXXX.mp4
        keyboard_scenarios/episode_XXXX/<filename>.mp4 + scenarios.json

    Returns the number of keyboard scenarios actually rolled out (0 if the
    episode had no usable chunks).
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
    # 1. Pull GT frames + actions from the dataset
    # ------------------------------------------------------------------
    gt_frame_chunks: list[np.ndarray] = []
    gt_action_chunks: list[np.ndarray] = []
    initial_frame: Optional[np.ndarray] = None

    for chunk_idx, ds_idx in enumerate(chunk_indices_gt):
        data = dataset[ds_idx]

        video = data["video"]
        if isinstance(video, torch.Tensor):
            video = video.permute(1, 2, 3, 0).numpy()  # (C,T,H,W) -> (T,H,W,C)
        elif video.ndim == 4 and video.shape[0] == 3:
            video = np.transpose(video, (1, 2, 3, 0))

        raw_actions = data["action"]
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.numpy()

        if chunk_idx == 0:
            initial_frame = video[0].copy()
            logger.info(
                f"  Video shape: {video.shape}, action dim: "
                f"{raw_actions.shape[-1]}D (raw) -> {MAX_ACTION_DIM}D (padded)"
            )

        gt_frame_chunks.append(video)
        gt_action_chunks.append(raw_actions)

    if initial_frame is None:
        logger.warning(f"Episode {episode_id}: no frames collected, skipping")
        return 0

    # Stitch GT video: keep the first chunk in full, drop frame 0 of every later chunk.
    gt_stitched = [gt_frame_chunks[0]]
    for c in gt_frame_chunks[1:]:
        gt_stitched.append(c[1:])
    gt_video = np.concatenate(gt_stitched, axis=0)

    # ------------------------------------------------------------------
    # 2. Save full GT
    # ------------------------------------------------------------------
    gt_path = os.path.join(save_dir, "gt", f"episode_{episode_id:04d}.mp4")
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    mediapy.write_video(gt_path, gt_video, fps=save_fps)
    logger.info(f"  Saved GT video ({len(gt_video)} frames) -> {gt_path}")

    # ------------------------------------------------------------------
    # 3. Autoregressive prediction with GT actions (faithful "best case")
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
    logger.info(f"  Saved predicted video ({len(predicted_video)} frames) -> {pred_path}")

    # Short reference clips (60-frame matches) for side-by-side comparison.
    _save_short_reference_videos(
        gt_video=gt_video,
        predicted_video=predicted_video,
        save_dir=save_dir,
        episode_id=episode_id,
        save_fps=save_fps,
    )

    # ------------------------------------------------------------------
    # 4. Keyboard-driven scenarios
    # ------------------------------------------------------------------
    scenarios_for_arm = _filter_scenarios_for_arm_layout(tuple(scenarios), arm_layout)
    if len(scenarios_for_arm) < len(scenarios):
        skipped = [s.filename for s in scenarios if s not in scenarios_for_arm]
        logger.info(
            f"  Single-arm embodiment: skipping right-arm-only scenarios: "
            f"{skipped}"
        )

    scenario_dir = os.path.join(save_dir, "keyboard_scenarios", f"episode_{episode_id:04d}")
    os.makedirs(scenario_dir, exist_ok=True)

    scenario_meta: list[dict[str, Any]] = []
    for scenario_idx, scenario in enumerate(scenarios_for_arm):
        trajectory = build_keyboard_trajectory(
            scenario=scenario,
            gains=gains,
            mean_action=mean_action,
            raw_dim=raw_dim,
            anchor_at_zero=anchor_at_zero,
        )
        padded_traj = pad_action(trajectory, MAX_ACTION_DIM)

        scenario_seed = seed + scenario_idx * 1000

        t0 = time.perf_counter()
        scenario_video = rollout_trajectory(
            video2world=video2world,
            initial_frame=initial_frame,
            padded_actions=padded_traj,
            num_chunks=NUM_OOD_CHUNKS,
            seed=scenario_seed,
            guidance=guidance,
        )
        elapsed = time.perf_counter() - t0

        scenario_path = os.path.join(scenario_dir, f"{scenario.filename}.mp4")
        mediapy.write_video(scenario_path, scenario_video, fps=save_fps)
        logger.info(
            f"  Scenario {scenario.filename}: {len(scenario_video)} frames, "
            f"{elapsed:.1f}s -- {scenario.description}"
        )

        scenario_meta.append({
            "filename": f"{scenario.filename}.mp4",
            "physical_keys": scenario.physical_keys,
            "description": scenario.description,
            "test": scenario.test,
            "segments": [
                {
                    "held_left": list(seg.held_left),
                    "held_right": list(seg.held_right),
                    "fast": seg.fast,
                    "slow": seg.slow,
                    "num_steps": seg.num_steps,
                }
                for seg in scenario.segments
            ],
            "frames": int(len(scenario_video)),
            "seed": int(scenario_seed),
            "inference_time_s": round(elapsed, 2),
        })

    meta_path = os.path.join(scenario_dir, "scenarios.json")
    with open(meta_path, "w") as f:
        json.dump({
            "episode_id": int(episode_id),
            "anchor": "zero" if anchor_at_zero else "empirical_mean",
            "gains": {
                "translation_xy": gains.translation_xy,
                "translation_z": gains.translation_z,
                "rotation": gains.rotation,
                "gripper": gains.gripper,
                "fast_multiplier": gains.fast_multiplier,
                "slow_multiplier": gains.slow_multiplier,
            },
            "arm_layout": arm_layout,
            "action_stats": {
                "raw_dim": int(raw_dim),
                "mean_l2": float(np.linalg.norm(mean_action)),
                "std_range": [float(std_action.min()), float(std_action.max())],
            },
            "scenarios": scenario_meta,
        }, f, indent=2)

    return len(scenarios_for_arm)


# ---------------------------------------------------------------------------
# Multi-dataset orchestration
# ---------------------------------------------------------------------------

def run_multi_dataset(
    args: argparse.Namespace,
    video2world: ActionVideo2WorldInference,
    gains: ControllerGains,
    scenarios: Sequence[KeyboardScenario],
) -> str:
    """Iterate the test_episodes.json dataset list and run each episode.

    Returns the absolute timestamped run directory.
    """
    with open(args.test_episodes_json, "r") as f:
        test_episodes: dict = json.load(f)

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_root = os.path.join(args.save_root, timestamp)
    os.makedirs(run_root, exist_ok=True)

    logger.info(f"Keyboard scenario generation -- saving to {run_root}")
    logger.info(
        f"Seed: {args.seed} | Episodes/dataset: {args.episodes_per_dataset} | "
        f"Anchor: {'zero' if args.anchor_at_zero else 'empirical_mean'} | "
        f"Stats episodes: {args.stats_episodes}"
    )
    logger.info(
        f"Gains: txy={gains.translation_xy} tz={gains.translation_z} "
        f"rot={gains.rotation} grip={gains.gripper} "
        f"fast={gains.fast_multiplier} slow={gains.slow_multiplier}"
    )
    logger.info(f"Scenarios ({len(scenarios)}): {[s.filename for s in scenarios]}")

    # Drop the README (legend) into the run dir up front so it's there even
    # if a long inference loop crashes mid-way.
    _write_readme(
        save_root=run_root,
        scenarios=scenarios,
        gains=gains,
        anchor_at_zero=args.anchor_at_zero,
        ckpt_path=args.ckpt_path,
        experiment=args.experiment,
    )

    excluded = set(args.exclude_datasets) if args.exclude_datasets else set()
    if excluded:
        logger.info(f"Excluding datasets: {sorted(excluded)}")

    run_meta: dict = {
        "timestamp": timestamp,
        "seed": args.seed,
        "episodes_per_dataset": args.episodes_per_dataset,
        "anchor": "zero" if args.anchor_at_zero else "empirical_mean",
        "stats_episodes": args.stats_episodes,
        "ckpt_path": args.ckpt_path,
        "experiment": args.experiment,
        "test_episodes_json": args.test_episodes_json,
        "exclude_datasets": sorted(excluded),
        "scenarios": [s.filename for s in scenarios],
        "gains": {
            "translation_xy": gains.translation_xy,
            "translation_z": gains.translation_z,
            "rotation": gains.rotation,
            "gripper": gains.gripper,
            "fast_multiplier": gains.fast_multiplier,
            "slow_multiplier": gains.slow_multiplier,
        },
        "datasets": {},
    }

    total_start = time.perf_counter()
    total_episodes = 0
    total_scenario_videos = 0

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

        # Keep only episodes that start at base_index=0 and span enough chunks.
        min_chunks_needed = max(NUM_OOD_CHUNKS, 2)
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
            f"(require >={min_chunks_needed} chunks)"
        )

        if not usable_episodes:
            logger.error(f"[{dataset_name}] No usable episodes found, skipping")
            continue

        # Action statistics (used for the per-step anchor when anchor_at_zero=False).
        mean_action, std_action, raw_dim, _ = compute_action_statistics(
            dataset, episode_map, timestep_interval,
            max_episodes=args.stats_episodes, seed=args.seed,
        )

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
            "raw_dim": int(raw_dim),
            "mean_l2": float(np.linalg.norm(mean_action)),
            "selected_episodes": selected,
        }

        for episode_id in selected:
            logger.info(f"[{dataset_name}] Processing episode {episode_id}")
            ep_start = time.perf_counter()
            try:
                n_scenarios = run_episode_kbd(
                    video2world=video2world,
                    dataset=dataset,
                    episode_map=episode_map,
                    episode_id=episode_id,
                    timestep_interval=timestep_interval,
                    arm_layout=arm_layout,
                    mean_action=mean_action,
                    std_action=std_action,
                    raw_dim=raw_dim,
                    save_dir=ds_save_dir,
                    seed=args.seed,
                    guidance=args.guidance,
                    save_fps=args.save_fps,
                    gains=gains,
                    anchor_at_zero=args.anchor_at_zero,
                    scenarios=scenarios,
                )
            except Exception as e:
                logger.error(f"[{dataset_name}] Error on episode {episode_id}: {e}")
                import traceback
                traceback.print_exc()
                raise

            ep_time = time.perf_counter() - ep_start
            if n_scenarios > 0:
                total_episodes += 1
                total_scenario_videos += n_scenarios
                logger.info(
                    f"[{dataset_name}] Episode {episode_id} done in {ep_time:.1f}s "
                    f"(GT + predicted + {n_scenarios} keyboard scenarios)"
                )
            else:
                logger.warning(f"[{dataset_name}] Episode {episode_id} skipped")

        run_meta["datasets"][dataset_name] = ds_meta

    meta_path = os.path.join(run_root, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    logger.info(f"Run metadata saved to {meta_path}")

    total_time = time.perf_counter() - total_start
    logger.info("=" * 70)
    logger.info(
        f"DONE -- {total_episodes} episodes, {total_scenario_videos} keyboard "
        f"scenario videos, {total_time:.1f}s total"
    )
    logger.info(f"Output: {run_root}")
    logger.info("=" * 70)
    return run_root


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse args, build gains + scenarios, load model, run."""
    torch.set_grad_enabled(False)
    args = parse_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gains = _build_gains_from_args(args)
    scenarios = _select_scenarios(args.scenarios)

    logger.info(f"Loading model from {args.ckpt_path}")
    video2world = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )
    if torch.cuda.is_available():
        mem_bytes = torch.cuda.memory_allocated()
        logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")

    run_multi_dataset(args=args, video2world=video2world, gains=gains, scenarios=scenarios)

    video2world.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
