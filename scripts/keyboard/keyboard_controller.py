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

"""Keyboard adapter -> Cosmos-H-Surgical-Simulator action chunks.

This module bridges a standard QWERTY keyboard to the 20-D dual-arm dVRK
kinematic action space that the Cosmos-H-Surgical-Simulator (C-H-S-S)
action-conditioned world model consumes.  It is the second control device
of the real-time simulator (the first being the Haply Inverse3 haptic
controller) and is intentionally simple so that anyone can drive the sim
with no extra hardware.

Action space recap (``jhu_dvrk_mono`` embodiment, see
``cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py``
and ``scripts/README_ACTION_SPACE.md``):

    Indices  Size  Contents                          Camera-frame side
    -------  ----  --------------------------------  -----------------
    0-2      3     PSM1 xyz_rel  (relative xyz)      RIGHT arm
    3-8      6     PSM1 rot6d_rel (6D rel rotation)  RIGHT arm
    9        1     PSM1 gripper  (absolute opening)  RIGHT arm
    10-12    3     PSM2 xyz_rel                      LEFT  arm
    13-18    6     PSM2 rot6d_rel                    LEFT  arm
    19       1     PSM2 gripper                      LEFT  arm
    20-43    24    zero-padding to MAX_ACTION_DIM=44 -

Values are in mean-std normalised space (mean ~= 0, std ~= 1 per dim) -
the same space the model was trained in.  Hence "do nothing" is the all-
zero vector and a held movement key adds a small constant offset (in
units of training-set standard deviations) for as long as the key is
held down.

The model consumes a chunk of 12 such vectors per inference call (one
context frame in, 12 predicted frames out at ~10 fps).  The controller
exposes :py:meth:`KeyboardController.get_action_chunk` which samples the
current key state and returns an ``(chunk_size, action_dim)`` array
ready to be passed to ``ActionVideo2WorldInference.step_inference``.

Key map (classic-gaming layout, two-handed control)
---------------------------------------------------

LEFT hand drives the LEFT (camera-frame) arm:
    W / S    y+ / y-          (move up / down in the image)
    A / D    x- / x+           (move left / right)
    R / F    z+ / z-           (push into / pull away from tissue)
    Q / E    yaw   - / +       (rot6d delta: dims 14 +-1/sqrt(2), 16 -/+1/sqrt(2))
    T / G    pitch + / -       (rot6d delta: dim 15 -/+)
    C / V    roll  - / +       (rot6d delta: dim 18 -/+)
    Space    gripper close (hold)
    Z        gripper open  (hold)   [Y on QWERTZ, see below]

RIGHT hand drives the RIGHT (camera-frame) arm:
    Up / Down arrow      y+ / y-
    Left / Right arrow   x- / x+
    I / K               z+ / z-
    U / O               yaw   - / +    (rot6d delta: dims 4, 6)
    J / L               pitch + / -    (rot6d delta: dim 5)
    N / M               roll  - / +    (rot6d delta: dim 8)
    Right Shift         gripper close (hold)
    Right Ctrl          gripper open  (hold)

Modifiers (apply to motion + rotation, NOT gripper):
    Left Shift   2.0x speed (fast)
    Left Ctrl    0.25x speed (precision)

Misc:
    P            print the current 20-D un-padded action snapshot
    Esc          request stop (sets ``stop_requested`` flag, listener exits)

Rotation mapping
----------------

Rot6d is the first two columns of the 3x3 rotation matrix flattened as
``[r00, r10, r20, r01, r11, r21]``.  Linearising R_axis(theta) at the
identity gives the small-angle Euler -> rot6d delta directions:

    yaw   (Z-axis): rot6d delta proportional to [0, +1, 0, -1, 0, 0]
    pitch (Y-axis): rot6d delta proportional to [0, 0, -1, 0, 0, 0]
    roll  (X-axis): rot6d delta proportional to [0, 0,  0, 0, 0, +1]

The yaw delta is unit-normalised (factor 1/sqrt(2) on each of dims 1 and
3) so all three axes produce rot6d perturbations of equal L2 norm for
the same gain.  This is the right linearisation for clean keyboard
control of small rotation deltas in the model's normalised action space;
for precise rotation control use the Haply Inverse3 instead.

Keyboard layouts (QWERTY vs QWERTZ)
-----------------------------------

The classic-gaming WASD layout is defined by *physical key position*,
not by the printed character.  On a QWERTZ keyboard (German, Swiss,
Czech, Slovak, Hungarian, ...) the physical keys at the QWERTY-Z and
QWERTY-Y positions produce the swapped characters ``y`` and ``z``
respectively.  Pass ``keyboard_layout="qwertz"`` (or ``--keyboard_layout
qwertz`` on the CLI) to swap every ``"z"`` <-> ``"y"`` character binding
so the same physical keys remain reachable from the WASD cluster:

    QWERTY: bottom-left key labelled Z -> char "z" -> gripper open (LEFT)
    QWERTZ: bottom-left key labelled Y -> char "y" -> gripper open (LEFT)

Non-character bindings (arrows, Shift, Space, Ctrl, Esc) are
layout-independent and unaffected.

Standalone usage
----------------

::

    python scripts/keyboard/keyboard_controller.py                       # 1 Hz, qwerty
    python scripts/keyboard/keyboard_controller.py --hz 10               # 10 Hz, qwerty
    python scripts/keyboard/keyboard_controller.py --keyboard_layout qwertz

Use the controller from your simulator like this::

    from scripts.keyboard.keyboard_controller import KeyboardController

    with KeyboardController() as kbd:
        while not kbd.stop_requested:
            action_chunk = kbd.get_action_chunk()              # shape (12, 44)
            next_frame, video = video2world.step_inference(
                img_array=current_frame,
                action=action_chunk.astype(np.float32),
                guidance=0,
                seed=seed,
                num_latent_conditional_frames=1,
            )
            display(video)
            current_frame = next_frame

Dependencies
------------

This module uses `pynput <https://pypi.org/project/pynput/>`_ for
cross-platform key event capture.  Install with::

    pip install pynput

On Linux it requires an X11/Wayland display (no headless terminals).
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pynput is required for keyboard_controller. Install with:\n"
        "    pip install pynput"
    ) from exc


# ---------------------------------------------------------------------------
# Action-space constants (mirror ``groot_configs.py`` / ``README_ACTION_SPACE.md``)
# ---------------------------------------------------------------------------

# Cosmos-H-Surgical-Simulator inference window.
CHUNK_SIZE = 12         # action timesteps per inference call (NUM_FRAMES - 1)
MAX_ACTION_DIM = 44     # zero-padded vector width (CMR Versius drives this)
DVRK_RAW_DIM = 20       # native un-padded width for jhu_dvrk_mono (and dVRK family)

# Per-arm slot offsets within the 20-D un-padded layout.
# (PSM1 = camera-frame RIGHT, PSM2 = camera-frame LEFT.)
_RIGHT_BASE = 0    # PSM1 starts at index 0
_LEFT_BASE = 10    # PSM2 starts at index 10
_XYZ_OFFSET = 0    # xyz occupies the first 3 dims of each per-arm block
_ROT6D_OFFSET = 3  # rot6d occupies the next 6 dims
_GRIPPER_OFFSET = 9  # gripper is the last (10th) dim of each per-arm block


# ---------------------------------------------------------------------------
# Default control gains (in units of training-set standard deviations)
# ---------------------------------------------------------------------------
# These are chosen to roughly match the OOD-scenario perturbation magnitudes
# in ``scripts/generate_open_h_ood_scenarios_depth.py`` (1.0sigma in-plane,
# 2.5sigma for depth).  Tune via ``ControllerGains`` if motion feels too
# slow/fast for a given checkpoint.
@dataclass
class ControllerGains:
    """Per-channel offsets applied per held key, in normalised (sigma) units."""

    translation_xy: float = 1.0   # x / y in-plane motion
    translation_z: float = 2.0    # z depth (smaller std in data -> needs more)
    rotation: float = 0.5         # rot6d perturbation per held rotation key
    gripper: float = 2.0          # absolute gripper target (held = open/close)
    fast_multiplier: float = 2.0  # Left-Shift modifier
    slow_multiplier: float = 0.25  # Left-Ctrl modifier


# ---------------------------------------------------------------------------
# Key bindings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Bind:
    """One keyboard -> action-dim mapping.

    A semantic key (e.g. ``"yaw_pos"``) may produce multiple ``_Bind`` entries
    when it needs to perturb several action dims at once (e.g. yaw rotation
    affects two rot6d dims simultaneously).
    """

    dim: int     # absolute index into the 20-D un-padded action vector
    coef: float  # signed multiplier in [-1, +1]; combined linearly with the gain
    kind: str    # one of {"translation", "rotation", "gripper"}


# Pre-computed rot6d delta coefficients for small-angle Euler-axis rotations.
#
# The rot6d basis is the first two columns of the 3x3 rotation matrix
# flattened as [r00, r10, r20, r01, r11, r21].  Linearising R_axis(theta) at
# theta = 0 (the action space's neutral / zero rotation) gives, for each
# axis, the rot6d delta direction:
#
#     X-axis (roll):  d/dtheta R_x|_0  -> rot6d delta = [0, 0, 0, 0, 0, +1]
#     Y-axis (pitch): d/dtheta R_y|_0  -> rot6d delta = [0, 0, -1, 0, 0, 0]
#     Z-axis (yaw):   d/dtheta R_z|_0  -> rot6d delta = [0, +1, 0, -1, 0, 0]
#
# We unit-normalise the yaw delta (factor 1/sqrt(2)) so all three axes
# produce rot6d perturbations of equal L2 norm for the same gain.  This
# means a "rotation" gain of 0.5 sigma always yields ||delta-rot6d|| = 0.5,
# regardless of which axis the user chose.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _arm_binds(base: int) -> dict[str, list[_Bind]]:
    """Build per-arm semantic-key -> list of action-dim binds.

    Returns a dict where each value is a *list* of :class:`_Bind` entries.
    Translation and gripper keys produce a single-element list; rotation
    keys may produce multiple entries to target the correct rot6d dims for
    the chosen Euler axis.
    """
    rot_base = base + _ROT6D_OFFSET
    return {
        # Translation (3 dims: x, y, z)
        "x_pos": [_Bind(base + _XYZ_OFFSET + 0, +1.0, "translation")],
        "x_neg": [_Bind(base + _XYZ_OFFSET + 0, -1.0, "translation")],
        "y_pos": [_Bind(base + _XYZ_OFFSET + 1, +1.0, "translation")],
        "y_neg": [_Bind(base + _XYZ_OFFSET + 1, -1.0, "translation")],
        "z_pos": [_Bind(base + _XYZ_OFFSET + 2, +1.0, "translation")],
        "z_neg": [_Bind(base + _XYZ_OFFSET + 2, -1.0, "translation")],
        # Yaw (Z-axis): perturb rot6d dims 1 (+) and 3 (-) by 1/sqrt(2)
        # so the L2 norm of the rot6d delta equals the gain magnitude.
        "yaw_pos":   [_Bind(rot_base + 1, +_INV_SQRT2, "rotation"),
                      _Bind(rot_base + 3, -_INV_SQRT2, "rotation")],
        "yaw_neg":   [_Bind(rot_base + 1, -_INV_SQRT2, "rotation"),
                      _Bind(rot_base + 3, +_INV_SQRT2, "rotation")],
        # Pitch (Y-axis): rot6d delta = [0, 0, -1, 0, 0, 0] -> dim 2 (-).
        "pitch_pos": [_Bind(rot_base + 2, -1.0, "rotation")],
        "pitch_neg": [_Bind(rot_base + 2, +1.0, "rotation")],
        # Roll (X-axis): rot6d delta = [0, 0, 0, 0, 0, +1] -> dim 5 (+).
        "roll_pos":  [_Bind(rot_base + 5, +1.0, "rotation")],
        "roll_neg":  [_Bind(rot_base + 5, -1.0, "rotation")],
        # Gripper (1 dim): absolute target, +ve = open, -ve = close.
        "grip_open":  [_Bind(base + _GRIPPER_OFFSET, +1.0, "gripper")],
        "grip_close": [_Bind(base + _GRIPPER_OFFSET, -1.0, "gripper")],
    }


# Semantic key -> physical pynput key for the LEFT arm (driven by left hand,
# WASD area).  String literals are character keys; pynput.keyboard.Key
# enums cover non-printables (Shift, Ctrl, Space, ...).
_LEFT_ARM_KEYMAP: dict[str, object] = {
    # Translation
    "y_pos":     "w",
    "y_neg":     "s",
    "x_neg":     "a",
    "x_pos":     "d",
    "z_pos":     "r",
    "z_neg":     "f",
    # Rotation
    "yaw_neg":   "q",
    "yaw_pos":   "e",
    "pitch_pos": "t",
    "pitch_neg": "g",
    "roll_neg":  "c",
    "roll_pos":  "v",
    # Gripper
    "grip_close": pynput_keyboard.Key.space,
    "grip_open":  "z",
}

# Semantic key -> physical pynput key for the RIGHT arm (driven by right
# hand, arrows / IJKL / numpad-ish area).
_RIGHT_ARM_KEYMAP: dict[str, object] = {
    # Translation
    "y_pos":     pynput_keyboard.Key.up,
    "y_neg":     pynput_keyboard.Key.down,
    "x_neg":     pynput_keyboard.Key.left,
    "x_pos":     pynput_keyboard.Key.right,
    "z_pos":     "i",
    "z_neg":     "k",
    # Rotation
    "yaw_neg":   "u",
    "yaw_pos":   "o",
    "pitch_pos": "j",
    "pitch_neg": "l",
    "roll_neg":  "n",
    "roll_pos":  "m",
    # Gripper
    "grip_close": pynput_keyboard.Key.shift_r,
    "grip_open":  pynput_keyboard.Key.ctrl_r,
}


# ---------------------------------------------------------------------------
# Internal pynput key normalisation
# ---------------------------------------------------------------------------

def _normalise_key(key) -> object:
    """Convert a pynput key event to a hashable token comparable to our keymaps.

    pynput returns:
      - ``KeyCode`` instances for character keys (with ``.char`` attr)
      - ``Key`` enum members for non-printables (Shift, Ctrl, Space, arrows, ...)

    We canonicalise character keys to their lowercase string and return the
    Key enum unchanged for non-printables.  Returns ``None`` for keys we
    can't interpret (e.g., dead keys, modifier-only events on some platforms).
    """
    if isinstance(key, pynput_keyboard.Key):
        return key
    if isinstance(key, pynput_keyboard.KeyCode):
        if key.char is None:
            return None
        return key.char.lower()
    return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

_VALID_LAYOUTS = ("qwerty", "qwertz")


@dataclass
class KeyboardControllerConfig:
    """Construction-time options for :class:`KeyboardController`.

    Attributes:
        chunk_size: Number of action timesteps returned by
            :meth:`KeyboardController.get_action_chunk`.  Must match the
            model's training horizon (12 for C-H-S-S).
        action_dim: Width of the returned action vector (44 for the
            unified Open-H 44-D space).  The 20 native dVRK channels are
            written and dims [20, action_dim) are zero-padded.
        gains: Per-channel sigma offsets (see :class:`ControllerGains`).
        keyboard_layout: ``"qwerty"`` (default) or ``"qwertz"``.  The
            classic-gaming WASD layout is defined by physical key position,
            not by the printed character on the key cap.  On QWERTZ
            keyboards (German, Swiss, Czech, Slovak, Hungarian, ...) the
            physical keys at the QWERTY-Z and QWERTY-Y positions produce
            the swapped characters ``y`` and ``z`` respectively.  Selecting
            ``"qwertz"`` swaps every ``"z"`` and ``"y"`` character binding
            so the same physical keys remain reachable from the WASD
            cluster.  Non-character bindings (arrows, Shift, Space, Ctrl,
            Esc) are layout-independent and unaffected.
        left_arm_keys: Override the LEFT-arm semantic-key -> physical-key
            map (advanced).  Applied AFTER the ``keyboard_layout`` swap.
        right_arm_keys: Override the RIGHT-arm semantic-key -> physical-key
            map (advanced).  Applied AFTER the ``keyboard_layout`` swap.
    """

    chunk_size: int = CHUNK_SIZE
    action_dim: int = MAX_ACTION_DIM
    gains: ControllerGains = field(default_factory=ControllerGains)
    keyboard_layout: str = "qwerty"
    left_arm_keys: dict[str, object] = field(
        default_factory=lambda: dict(_LEFT_ARM_KEYMAP)
    )
    right_arm_keys: dict[str, object] = field(
        default_factory=lambda: dict(_RIGHT_ARM_KEYMAP)
    )

    def __post_init__(self) -> None:
        if self.keyboard_layout not in _VALID_LAYOUTS:
            raise ValueError(
                f"Unknown keyboard_layout {self.keyboard_layout!r}; "
                f"must be one of {_VALID_LAYOUTS}"
            )
        if self.keyboard_layout == "qwertz":
            # On QWERTZ, the "Y" and "Z" physical keys are swapped relative
            # to QWERTY.  Mirror that swap in any character bindings so
            # WASD-cluster ergonomics are preserved.  Non-string (Key.*)
            # entries are left untouched.
            self.left_arm_keys = _swap_yz_chars(self.left_arm_keys)
            self.right_arm_keys = _swap_yz_chars(self.right_arm_keys)


def _swap_yz_chars(keymap: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``keymap`` with every ``"y"`` <-> ``"z"`` char swapped."""
    out: dict[str, object] = {}
    for sem, phys in keymap.items():
        if phys == "z":
            out[sem] = "y"
        elif phys == "y":
            out[sem] = "z"
        else:
            out[sem] = phys
    return out


class KeyboardController:
    """Adapter from live keyboard input to C-H-S-S action chunks.

    Spawns a `pynput` listener thread that maintains a thread-safe set of
    currently-held keys.  Each call to :meth:`get_action_chunk` snapshots
    that set, builds a 20-D un-padded action vector in normalised
    (mean/std) space according to the configured key bindings and gains,
    and tiles it into a ``(chunk_size, action_dim)`` array padded with
    zeros to the unified Cosmos action width.

    The controller is stateless across calls: there is no integration of
    held keys over time, only a per-call snapshot.  This matches the
    open-loop "feed 12 normalised actions, get 12 frames" interface that
    Cosmos-Predict2.5 expects, and keeps the controller decoupled from
    whatever rate the simulator main loop runs at.

    Lifecycle (preferred form is the context manager)::

        with KeyboardController() as kbd:
            while not kbd.stop_requested:
                action = kbd.get_action_chunk()
                ...

    Manual lifecycle (be sure to call ``stop`` to release the listener)::

        kbd = KeyboardController()
        kbd.start()
        try:
            ...
        finally:
            kbd.stop()
    """

    def __init__(self, config: Optional[KeyboardControllerConfig] = None):
        self.config = config or KeyboardControllerConfig()
        if self.config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.config.chunk_size}")
        if self.config.action_dim < DVRK_RAW_DIM:
            raise ValueError(
                f"action_dim={self.config.action_dim} is smaller than the "
                f"native dVRK width ({DVRK_RAW_DIM}); cannot fit per-arm channels"
            )

        # Build the per-arm semantic -> [_Bind...] mappings ONCE so the hot
        # path is just a dict lookup.  Each semantic key may produce
        # multiple binds (rotation keys perturb >1 rot6d dim).
        self._left_binds: dict[str, list[_Bind]] = _arm_binds(_LEFT_BASE)
        self._right_binds: dict[str, list[_Bind]] = _arm_binds(_RIGHT_BASE)

        # Reverse-lookup: physical key token -> flat list[_Bind].  Multiple
        # arms / semantic keys can share a physical key (e.g. if a user
        # custom-binds them) and a single semantic key can map to multiple
        # action dims (e.g. yaw -> two rot6d dims) - both cases collapse
        # into the same flat list.
        self._key_to_binds: dict[object, list[_Bind]] = {}
        for sem, phys in self.config.left_arm_keys.items():
            if sem not in self._left_binds:
                raise ValueError(f"Unknown LEFT-arm semantic key '{sem}'")
            for bind in self._left_binds[sem]:
                self._key_to_binds.setdefault(phys, []).append(bind)
        for sem, phys in self.config.right_arm_keys.items():
            if sem not in self._right_binds:
                raise ValueError(f"Unknown RIGHT-arm semantic key '{sem}'")
            for bind in self._right_binds[sem]:
                self._key_to_binds.setdefault(phys, []).append(bind)

        # Modifier and special-purpose keys.
        self._fast_modifier = pynput_keyboard.Key.shift
        self._slow_modifier = pynput_keyboard.Key.ctrl
        self._print_key = "p"
        self._stop_key = pynput_keyboard.Key.esc

        # Mutable state guarded by ``_lock``.
        self._lock = threading.Lock()
        self._held_keys: set[object] = set()
        self._listener: Optional[pynput_keyboard.Listener] = None
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "KeyboardController":
        """Spawn the pynput listener thread (idempotent)."""
        if self._listener is not None:
            return self
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.daemon = True
        self._listener.start()
        return self

    def stop(self) -> None:
        """Stop the listener thread and clear held-key state."""
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.stop()
        with self._lock:
            self._held_keys.clear()

    def __enter__(self) -> "KeyboardController":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    @property
    def stop_requested(self) -> bool:
        """``True`` once the user has pressed Esc (or the stop binding)."""
        return self._stop_requested

    # ------------------------------------------------------------------
    # pynput callbacks (run on the listener thread)
    # ------------------------------------------------------------------

    def _on_press(self, key) -> Optional[bool]:
        token = _normalise_key(key)
        if token is None:
            return None
        if token == self._stop_key:
            self._stop_requested = True
            return False  # tells pynput to stop the listener
        with self._lock:
            self._held_keys.add(token)
            # Modifier "pair" canonicalisation: pynput emits shift_l/shift_r and
            # ctrl_l/ctrl_r as distinct keys.  Mirror them under the generic
            # Key.shift / Key.ctrl tokens so the modifier check below sees them.
            if token in (pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r):
                self._held_keys.add(pynput_keyboard.Key.shift)
            elif token in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                self._held_keys.add(pynput_keyboard.Key.ctrl)
        if token == self._print_key:
            self._print_snapshot()
        return None

    def _on_release(self, key) -> None:
        token = _normalise_key(key)
        if token is None:
            return
        with self._lock:
            self._held_keys.discard(token)
            # Mirror modifier pairs: only clear the generic alias when BOTH
            # left/right variants are released.
            if token in (pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r):
                if (pynput_keyboard.Key.shift_l not in self._held_keys
                        and pynput_keyboard.Key.shift_r not in self._held_keys):
                    self._held_keys.discard(pynput_keyboard.Key.shift)
            elif token in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r):
                if (pynput_keyboard.Key.ctrl_l not in self._held_keys
                        and pynput_keyboard.Key.ctrl_r not in self._held_keys):
                    self._held_keys.discard(pynput_keyboard.Key.ctrl)

    # ------------------------------------------------------------------
    # Action sampling (called by the simulator main loop)
    # ------------------------------------------------------------------

    def _build_action_step(self) -> np.ndarray:
        """Compose the un-padded 20-D action for the current key state.

        Returns:
            ``(DVRK_RAW_DIM,)`` ``float32`` vector in normalised space.
            Translation / rotation channels are signed sigma offsets;
            gripper channels are absolute targets that decay back to 0
            (= mean opening) when no gripper key is held.
        """
        gains = self.config.gains
        with self._lock:
            held = frozenset(self._held_keys)

        # Determine modifier scaling for non-gripper channels.
        modifier = 1.0
        if self._fast_modifier in held:
            modifier *= gains.fast_multiplier
        if self._slow_modifier in held:
            modifier *= gains.slow_multiplier

        action = np.zeros(DVRK_RAW_DIM, dtype=np.float32)

        for token in held:
            binds = self._key_to_binds.get(token)
            if not binds:
                continue
            for bind in binds:
                if bind.kind == "translation":
                    # Pick translation_xy vs translation_z based on the
                    # within-arm slot (xyz occupies dims 0/1/2 of each
                    # 10-D per-arm block, so dim % 10 == 2 is the z axis).
                    if bind.dim % 10 == _XYZ_OFFSET + 2:
                        gain = gains.translation_z
                    else:
                        gain = gains.translation_xy
                    action[bind.dim] += bind.coef * gain * modifier
                elif bind.kind == "rotation":
                    action[bind.dim] += bind.coef * gains.rotation * modifier
                elif bind.kind == "gripper":
                    # Absolute target, NOT scaled by speed modifiers.
                    # If both open and close are pressed, sum to ~0.
                    action[bind.dim] += bind.coef * gains.gripper

        return action

    def get_action_chunk(self) -> np.ndarray:
        """Return the next ``(chunk_size, action_dim)`` action chunk.

        The chunk is built by snapshotting the current key state once and
        tiling the resulting 20-D step to ``chunk_size`` rows; the last
        ``action_dim - 20`` columns are zero-padded.  This produces a
        constant action over the upcoming inference horizon, which is the
        natural interpretation of "while I'm holding W, the left arm
        should keep moving up for the next 1.2 s".

        Returns:
            ``np.ndarray`` of shape ``(chunk_size, action_dim)``,
            dtype ``float32``, in mean-std normalised space - directly
            consumable by ``ActionVideo2WorldInference.step_inference``.
        """
        step = self._build_action_step()
        out = np.zeros((self.config.chunk_size, self.config.action_dim), dtype=np.float32)
        out[:, :DVRK_RAW_DIM] = step
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_snapshot(self) -> None:
        """Print the current 20-D un-padded action vector (for debugging)."""
        step = self._build_action_step()
        right = step[_RIGHT_BASE:_RIGHT_BASE + 10]
        left = step[_LEFT_BASE:_LEFT_BASE + 10]
        print(
            "[keyboard_controller] snapshot:\n"
            f"  RIGHT (PSM1, dims 0-9):  xyz={_fmt_vec(right[:3])}  "
            f"rot6d={_fmt_vec(right[3:9])}  grip={right[9]:+.3f}\n"
            f"  LEFT  (PSM2, dims 10-19): xyz={_fmt_vec(left[:3])}  "
            f"rot6d={_fmt_vec(left[3:9])}  grip={left[9]:+.3f}",
            flush=True,
        )

    def describe_keymap(self) -> str:
        """Human-readable summary of the active key bindings."""
        # The bottom-left WASD-cluster key is "Z" on QWERTY but "Y" on
        # QWERTZ - keep the documentation in sync with the actual binding.
        grip_open_left = "Y" if self.config.keyboard_layout == "qwertz" else "Z"
        layout_note = ""
        if self.config.keyboard_layout == "qwertz":
            layout_note = (
                " [QWERTZ: 'Z' / 'Y' character bindings have been swapped so "
                "the physical WASD-cluster key still triggers gripper open]"
            )
        lines = [
            f"Cosmos-H-Surgical-Simulator keyboard controller key map "
            f"(layout = {self.config.keyboard_layout}){layout_note}",
            "  LEFT arm  (camera-frame left  / PSM2, dims 10-19):",
            "    W / S         : y +/- (image up/down)",
            "    A / D         : x -/+ (image left/right)",
            "    R / F         : z +/- (push/pull, depth)",
            "    Q / E         : yaw  -/+    [rot6d dims 14, 16]",
            "    T / G         : pitch +/-   [rot6d dim  15]",
            "    C / V         : roll  -/+   [rot6d dim  18]",
            "    Space         : gripper close (hold)",
            f"    {grip_open_left}             : gripper open  (hold)",
            "  RIGHT arm (camera-frame right / PSM1, dims 0-9):",
            "    Up / Down     : y +/-",
            "    Left / Right  : x -/+",
            "    I / K         : z +/-",
            "    U / O         : yaw  -/+    [rot6d dims  4,  6]",
            "    J / L         : pitch +/-   [rot6d dim   5]",
            "    N / M         : roll  -/+   [rot6d dim   8]",
            "    R-Shift       : gripper close (hold)",
            "    R-Ctrl        : gripper open  (hold)",
            "  Modifiers (apply to motion + rotation, NOT gripper):",
            f"    L-Shift       : x{self.config.gains.fast_multiplier:g} speed",
            f"    L-Ctrl        : x{self.config.gains.slow_multiplier:g} speed (precision)",
            "  Misc:",
            "    P             : print current 20-D action snapshot",
            "    Esc           : stop / exit",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stand-alone CLI: print live action chunks at a fixed rate
# ---------------------------------------------------------------------------

def _fmt_vec(v: np.ndarray) -> str:
    """Compact, fixed-width tuple format for an action sub-vector."""
    return "(" + ",".join(f"{x:+5.2f}" for x in v) + ")"


def _format_action_row(row: np.ndarray) -> str:
    """Two-line preview of a 20-D un-padded action row (for the CLI).

    Earlier versions collapsed rot6d into a single L2-norm scalar, which
    made all rotation keys look identical in the CLI output (every single
    rotation key produces the same rot-norm).  We now print the full 6-D
    rot6d delta so yaw / pitch / roll / sign differences are visible.
    """
    right = row[_RIGHT_BASE:_RIGHT_BASE + 10]
    left = row[_LEFT_BASE:_LEFT_BASE + 10]
    return (
        f"R xyz={_fmt_vec(right[:3])}  r6d={_fmt_vec(right[3:9])}  g={right[9]:+5.2f}\n"
        f"  L xyz={_fmt_vec(left[:3])}  r6d={_fmt_vec(left[3:9])}  g={left[9]:+5.2f}"
    )


def _run_cli_demo(
    hz: float = 1.0,
    action_dim: int = MAX_ACTION_DIM,
    keyboard_layout: str = "qwerty",
) -> None:
    """Spin a polling loop that prints the current action chunk at ``hz``.

    Intended as a sanity-check / smoke test that the listener is alive,
    that key bindings produce the expected dim/coef offsets, and that
    Esc cleanly tears the listener down.  Drives the same code path that
    a real simulator would: ``get_action_chunk()`` -> ``(12, action_dim)``
    -> log first row.
    """
    cfg = KeyboardControllerConfig(
        action_dim=action_dim,
        keyboard_layout=keyboard_layout,
    )
    print(
        f"action_dim = {cfg.action_dim}, chunk_size = {cfg.chunk_size}, "
        f"keyboard_layout = {cfg.keyboard_layout}"
    )
    print()
    with KeyboardController(cfg) as kbd:
        print(kbd.describe_keymap())
        print()
        print(
            "Streaming current action chunk at "
            f"{hz:g} Hz; press Esc to quit.\n"
            "Each entry shows the FIRST row of the (chunk_size, action_dim) "
            "tile (all rows are identical for a held-key snapshot).\n"
            "  xyz = (x, y, z) translation in sigma units\n"
            "  r6d = full 6-D rot6d delta in sigma units (axis-aligned: "
            "yaw -> dims 1,3; pitch -> dim 2; roll -> dim 5)\n"
            "  g   = absolute gripper target in sigma units"
        )
        period = 1.0 / max(hz, 1e-6)
        try:
            while not kbd.stop_requested:
                t0 = time.perf_counter()
                chunk = kbd.get_action_chunk()
                # Quick shape sanity-check on every iteration; cheap.
                assert chunk.shape == (cfg.chunk_size, cfg.action_dim)
                assert chunk.dtype == np.float32
                first_row = chunk[0]
                print(
                    f"chunk[0,:20]: {_format_action_row(first_row)}\n"
                    f"  pad sum = {first_row[DVRK_RAW_DIM:].sum():.0f}",
                    flush=True,
                )
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, period - elapsed))
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt - exiting")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stand-alone smoke test of the C-H-S-S keyboard controller.",
    )
    parser.add_argument(
        "--hz", type=float, default=1.0,
        help="Action-chunk polling rate (default: 1 Hz)",
    )
    parser.add_argument(
        "--action_dim", type=int, default=MAX_ACTION_DIM,
        help=(
            "Width of the returned action vector "
            f"(default: {MAX_ACTION_DIM} = MAX_ACTION_DIM; the native dVRK "
            f"native width is {DVRK_RAW_DIM})"
        ),
    )
    parser.add_argument(
        "--keyboard_layout", type=str, default="qwerty",
        choices=list(_VALID_LAYOUTS),
        help=(
            "Physical keyboard layout (default: qwerty).  Use 'qwertz' on "
            "German / Swiss / Czech / Slovak / Hungarian keyboards so the "
            "WASD-cluster gripper-open key remains in the bottom-left "
            "position (which produces 'y' instead of 'z' on QWERTZ)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    _run_cli_demo(
        hz=args.hz,
        action_dim=args.action_dim,
        keyboard_layout=args.keyboard_layout,
    )


if __name__ == "__main__":
    main()
