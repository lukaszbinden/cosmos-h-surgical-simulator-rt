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
    Right Shift         gripper close (latched -- see Gripper mode)
    Right Ctrl          gripper open  (latched -- see Gripper mode)

Gripper mode (``KeyboardControllerConfig.gripper_mode``, default ``"latch"``):
    The gripper channels (dims 9 / 19) are *absolute* opening targets
    in sigma units, unlike the *relative* xyz / rot6d channels.  A
    "do nothing this frame" gripper command is therefore NOT 0 sigma
    -- 0 sigma maps to the dataset-mean opening, so emitting 0 actively
    drives the jaws to mid-position.  In ``"latch"`` mode each gripper
    key press updates a per-arm sticky state in {open, closed} that is
    re-emitted on every action step until the opposite gripper key on
    that arm is pressed (or F1 resets the scene), so e.g. pressing
    Right-Ctrl once + holding Up-arrow keeps the right gripper open
    while the arm moves up.  In ``"momentary"`` mode the legacy held-
    only behaviour is used (release the gripper key -> channel returns
    to 0 sigma -> jaws snap to mean).

Speed levels (sticky, apply to motion + rotation, NOT gripper):
    Alt+1 .. Alt+4   Switch the per-held-key speed multiplier between 4
                     user-configurable scalars (defaults: 0.25x / 1x /
                     2x / 4x for precision / normal / fast / turbo).
                     The selected level persists until the next Alt+digit
                     press; startup default is configurable
                     (:attr:`ControllerGains.default_speed_level`,
                     defaults to ``2`` -> 1.0x).  Left and right Alt are
                     both accepted; on QWERTZ the right-Alt position is
                     usually AltGr (a separate key emitted as
                     ``Key.alt_gr``) and will not change speed.

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

# pynput is the cross-platform key-event capture library used by the live
# controller.  On headless compute nodes (no X11/Wayland display) pynput's
# *import-time* backend selection raises ImportError, even when the package
# itself is installed -- the failure happens deep inside
# ``pynput/_util/__init__.py::backend()``.
#
# We catch that here and substitute a minimal stub for the ``pynput.keyboard``
# namespace so the rest of this module can still load.  This lets *offline*
# callers -- which only need the action-construction primitives such as
# :class:`ControllerGains`, :func:`_arm_binds`, and the index constants --
# import :mod:`scripts.keyboard.keyboard_controller` without an X server.
#
# The :class:`KeyboardController` class itself is unaffected for the live use
# case: when a real pynput is available the original module loads as before.
# When the stub is in use, instantiating the controller raises a helpful
# ImportError (via the stub Listener) so the live use case fails loudly with
# a clear message rather than silently doing nothing.
try:
    from pynput import keyboard as pynput_keyboard
    _PYNPUT_AVAILABLE = True
    _PYNPUT_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _pynput_exc:  # noqa: BLE001 - includes display-backend errors
    import types as _types
    _PYNPUT_AVAILABLE = False
    _PYNPUT_IMPORT_ERROR = _pynput_exc

    # Names referenced from the per-arm keymaps below; values just need to be
    # distinct, hashable sentinels so the ``_key_to_binds`` dict keeps separate
    # entries per stubbed key (important if any test code paths walk the dict
    # without instantiating the controller).
    _PYNPUT_KEY_NAMES = (
        "space", "up", "down", "left", "right",
        "shift", "ctrl", "shift_l", "shift_r", "ctrl_l", "ctrl_r", "esc",
        # Alt is the live controller's sticky-speed modifier (Alt+1..Alt+N
        # swaps :attr:`ControllerGains.speed_levels`); ``alt_gr`` is listed
        # for completeness so the headless import path doesn't crash if any
        # caller introspects it (we deliberately do NOT bind AltGr).
        "alt", "alt_l", "alt_r", "alt_gr",
        # F1 = scene-reset binding (see ``KeyboardController.reset_requested``);
        # listed here so the offline ``_StubListener`` import path keeps a
        # distinct sentinel for the key.
        "f1",
    )

    class _StubListener:
        """Stand-in for ``pynput.keyboard.Listener`` when pynput cannot load.

        Raising at instantiation time -- rather than at module import -- is
        deliberate: it lets the offline imports succeed while still failing
        loudly the moment a caller tries to spin up live keyboard capture.
        """

        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "Live KeyboardController needs pynput, but pynput's display-backend "
                f"selection failed at module load time: {_PYNPUT_IMPORT_ERROR}.  "
                "Live keyboard control requires an X11/Wayland display; the offline "
                "action-construction primitives (ControllerGains, _arm_binds, "
                "DVRK_RAW_DIM, ...) remain importable on headless hosts."
            )

        # The live controller calls .start()/.stop() on the listener; provide
        # cheap no-ops so __exit__ paths after a failed start don't raise twice.
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    pynput_keyboard = _types.SimpleNamespace(
        Key=_types.SimpleNamespace(
            **{n: f"<pynput.Key.{n}-stub>" for n in _PYNPUT_KEY_NAMES}
        ),
        # KeyCode is referenced inside _normalise_key for an isinstance check,
        # which is only reached on a live listener callback.  An empty class
        # is sufficient: nothing will ever be an instance of it on a headless
        # host because no events are produced.
        KeyCode=type("KeyCodeStub", (), {}),
        Listener=_StubListener,
    )


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
#
# Rotation gain bumped from 0.5 to 1.0sigma after qualitative evaluation of
# the iter_4k anneal teacher: at 0.5sigma the rendered yaw / pitch / roll
# response was barely visible (essentially indistinguishable from no
# rotation), while at 1.0sigma the response is clearly present.  1.0sigma
# also keeps the per-axis rot6d delta L2 norm at parity with the in-plane
# translation gain, which is the right default for one-rotation-key-press
# control fidelity in live use.
@dataclass
class ControllerGains:
    """Per-channel offsets applied per held key, in normalised (sigma) units.

    Speed scaling
    -------------

    The live :class:`KeyboardController` exposes 4 sticky speed levels
    selected by ``Alt+1`` .. ``Alt+4``; pressing one of these
    combinations swaps the active scalar in :attr:`speed_levels` and the
    new scalar persists until the next ``Alt+digit`` press.  This
    replaces the older "hold L-Shift for fast / hold L-Ctrl for slow"
    scheme so the only Shift / Ctrl bindings left in the live controller
    are the gripper keys (R-Shift / R-Ctrl).

    The legacy :attr:`fast_multiplier` / :attr:`slow_multiplier` fields
    are retained ONLY for the offline OOD scenario generator at
    ``scripts/generate_keyboard_input_scenarios.py``, which bakes "fast"
    or "slow" segments into pre-recorded ``actions.npy`` files.  The
    live controller ignores them.
    """

    translation_xy: float = 1.0   # x / y in-plane motion
    translation_z: float = 2.0    # z depth (smaller std in data -> needs more)
    rotation: float = 1.0         # rot6d perturbation per held rotation key
    gripper: float = 2.0          # absolute gripper target (held = open/close)

    # Live-controller speed presets selected by Alt+1 .. Alt+N.  Length
    # determines how many Alt+digit slots are bound (capped at 9 for
    # Alt+1..Alt+9).  Defaults give the user precision / normal / fast /
    # turbo at the canonical 0.25x / 1x / 2x / 4x ratios.
    speed_levels: tuple[float, ...] = (0.25, 1.0, 2.0, 4.0)
    # 1-indexed initial level applied at startup, before any Alt+digit
    # press.  Default ``2`` -> 1.0x in the default speed_levels.
    default_speed_level: int = 2

    # Deprecated -- only consumed by ``scripts/generate_keyboard_input_scenarios.py``.
    fast_multiplier: float = 2.0
    slow_multiplier: float = 0.25

    def __post_init__(self) -> None:
        n = len(self.speed_levels)
        if n == 0:
            raise ValueError("speed_levels must contain at least one scalar")
        if n > 9:
            raise ValueError(
                f"speed_levels supports at most 9 entries (Alt+1..Alt+9); got {n}"
            )
        if not 1 <= self.default_speed_level <= n:
            raise ValueError(
                f"default_speed_level={self.default_speed_level} out of range "
                f"[1, {n}] for speed_levels={self.speed_levels}"
            )


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

    Sign convention -- IMPORTANT for x-axis:

    The semantic keys ``x_pos`` / ``x_neg`` describe *image-frame* intent
    (``x_pos`` = "move tool to the right in the rendered video";
    ``x_neg`` = "move tool to the left").  This matches the user-facing
    documentation and the WASD-style mental model.

    For the JHU dVRK mono training data the underlying ``action`` channel
    on the corresponding xyz dim is in the **PSM kinematic frame**, which
    is *mirrored* relative to the camera frame for both PSMs.  Empirically
    (verified on the iter_4k anneal teacher at 1.0sigma and 2.0sigma),
    pressing D / Right-arrow with the original ``+1.0`` coef produced
    image-LEFT motion, and A / Left-arrow with ``-1.0`` produced image-
    right motion -- the opposite of what the keymap docstring promises.

    We therefore flip the x sign at the bind layer so that the *semantic*
    intent ("x_pos = image right") translates to the correct *kinematic*
    sign on the action dim.  This change is transparent to callers: every
    consumer of ``_arm_binds`` sees ``x_pos`` and ``x_neg`` produce the
    intuitive image-frame motion direction, without needing to know about
    the dataset's kinematic-frame convention.

    The y and z axes are unaffected -- both are correctly aligned with
    the image frame in the data (``+y`` = image up, ``+z`` = into tissue),
    so ``y_pos`` / ``y_neg`` and ``z_pos`` / ``z_neg`` keep their
    straightforward signs.
    """
    rot_base = base + _ROT6D_OFFSET
    return {
        # Translation (3 dims: x, y, z).
        # NOTE: x is sign-flipped here; see the function docstring above.
        "x_pos": [_Bind(base + _XYZ_OFFSET + 0, -1.0, "translation")],
        "x_neg": [_Bind(base + _XYZ_OFFSET + 0, +1.0, "translation")],
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


# Display ordering for semantic command names.  Used by
# :py:meth:`KeyboardController.get_active_commands` so callers can render a
# HUD overlay with a stable, intuitive ordering (translation -> rotation ->
# gripper) regardless of dict-iteration order.  Keep in sync with
# :func:`_arm_binds`'s set of semantic keys.
_SEMANTIC_DISPLAY_ORDER: tuple[str, ...] = (
    "x_pos", "x_neg",
    "y_pos", "y_neg",
    "z_pos", "z_neg",
    "yaw_pos", "yaw_neg",
    "pitch_pos", "pitch_neg",
    "roll_pos", "roll_neg",
    "grip_open", "grip_close",
)


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
_VALID_GRIPPER_MODES = ("latch", "momentary")


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
        gripper_mode: ``"latch"`` (default) or ``"momentary"``.  Controls
            how the absolute gripper channels (dims 9 / 19) are driven:

            * ``"latch"``: each gripper key press updates a per-arm sticky
              state in {open, closed} that persists until the opposite
              gripper key is pressed (or F1 resets the scene).  The
              gripper sigma value is then re-emitted on every action
              step regardless of whether any key is held.  This is the
              correct behaviour for the JHU dVRK action space because
              ``gripper`` is an *absolute* opening target and 0 sigma
              maps to the dataset-mean opening, not to "do nothing";
              releasing the gripper key in momentary mode therefore
              snaps the jaws back to the mean.  Per-arm states default
              to ``neutral`` (gripper = 0 sigma) at startup and on F1.
            * ``"momentary"``: legacy held-only behaviour -- gripper
              sigma is +/- gain ONLY while the open / close key is
              physically held; release a key and the channel returns to
              0 sigma, which the model interprets as "drive the jaws to
              the mean opening".  Kept for backward compatibility.
        left_arm_keys: Override the LEFT-arm semantic-key -> physical-key
            map (advanced).  Applied AFTER the ``keyboard_layout`` swap.
        right_arm_keys: Override the RIGHT-arm semantic-key -> physical-key
            map (advanced).  Applied AFTER the ``keyboard_layout`` swap.
    """

    chunk_size: int = CHUNK_SIZE
    action_dim: int = MAX_ACTION_DIM
    gains: ControllerGains = field(default_factory=ControllerGains)
    keyboard_layout: str = "qwerty"
    gripper_mode: str = "latch"
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
        if self.gripper_mode not in _VALID_GRIPPER_MODES:
            raise ValueError(
                f"Unknown gripper_mode {self.gripper_mode!r}; "
                f"must be one of {_VALID_GRIPPER_MODES}"
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
        # Sticky-gripper reverse-lookup: physical key -> ("left" | "right",
        # +1 (open) | -1 (close)).  Populated from grip_open / grip_close
        # entries in each arm's keymap so ``_on_press`` can update the
        # latched per-arm state in O(1).  Non-gripper keys are absent.
        self._gripper_keys: dict[object, tuple[str, int]] = {}
        for sem, phys in self.config.left_arm_keys.items():
            if sem not in self._left_binds:
                raise ValueError(f"Unknown LEFT-arm semantic key '{sem}'")
            for bind in self._left_binds[sem]:
                self._key_to_binds.setdefault(phys, []).append(bind)
            if sem == "grip_open":
                self._gripper_keys[phys] = ("left", +1)
            elif sem == "grip_close":
                self._gripper_keys[phys] = ("left", -1)
        for sem, phys in self.config.right_arm_keys.items():
            if sem not in self._right_binds:
                raise ValueError(f"Unknown RIGHT-arm semantic key '{sem}'")
            for bind in self._right_binds[sem]:
                self._key_to_binds.setdefault(phys, []).append(bind)
            if sem == "grip_open":
                self._gripper_keys[phys] = ("right", +1)
            elif sem == "grip_close":
                self._gripper_keys[phys] = ("right", -1)

        # Special-purpose keys.
        self._print_key = "p"
        self._stop_key = pynput_keyboard.Key.esc
        # Scene-reset binding -- F1.  Setting this raises the
        # ``reset_requested`` flag (see the property below); callers consume
        # it via ``consume_reset_request`` to atomically read-and-clear,
        # which lets them e.g. rebuild a streaming inference KV cache from
        # the original first frame to recover from accumulated drift.
        # Unlike ``Esc``, F1 does NOT terminate the listener -- the user
        # can keep driving the sim immediately after the reset completes.
        self._reset_key = pynput_keyboard.Key.f1

        # Sticky speed selector -- ``Alt + digit`` swaps the active scalar
        # in :attr:`ControllerGains.speed_levels`.  We pre-build the
        # digit-char -> 0-indexed level map here so the listener hot path
        # is just a dict lookup and an ``if alt in held``.  Length matches
        # the configured ``speed_levels`` (capped at 9 by ``__post_init__``).
        _digit_chars = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
        n_levels = len(self.config.gains.speed_levels)
        self._speed_digit_keys: dict[str, int] = {
            _digit_chars[i]: i for i in range(n_levels)
        }
        # Mirror left/right Alt under this generic alias so the Alt+digit
        # check below can do a single membership test regardless of which
        # physical Alt key the user pressed.  See ``_on_press``.
        self._alt_alias = pynput_keyboard.Key.alt

        # Mutable state guarded by ``_lock``.
        self._lock = threading.Lock()
        self._held_keys: set[object] = set()
        self._listener: Optional[pynput_keyboard.Listener] = None
        self._stop_requested = False
        self._reset_requested = False
        # 0-indexed active speed level + cached scalar.  Both mutated by
        # ``_on_press`` (listener thread), read by ``_build_action_step``
        # (caller thread) under ``_lock``.
        self._current_speed_level: int = self.config.gains.default_speed_level - 1
        self._current_speed_scalar: float = float(
            self.config.gains.speed_levels[self._current_speed_level]
        )
        # Per-arm latched gripper state in {-1 (closed), 0 (neutral), +1
        # (open)}.  Updated by ``_on_press`` whenever a gripper key on
        # that arm is pressed; cleared back to 0 on F1 (scene reset).
        # Read by ``_build_action_step`` only when ``gripper_mode ==
        # "latch"`` -- in ``"momentary"`` mode the legacy held-only
        # gripper logic runs and these values are bookkeeping-only.
        self._left_grip_state: int = 0
        self._right_grip_state: int = 0

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

    @property
    def reset_requested(self) -> bool:
        """``True`` once the user has pressed F1 since the last consume.

        The flag is *sticky*: it stays ``True`` until the caller resets it
        via :py:meth:`consume_reset_request` (which atomically reads and
        clears).  This lets a slow consumer (e.g. a per-AR-step inference
        loop) miss at most one reset between polls but never lose one.
        """
        return self._reset_requested

    def consume_reset_request(self) -> bool:
        """Atomically read-and-clear :py:attr:`reset_requested`.

        Returns ``True`` if a reset had been requested since the last call,
        and resets the flag so the next call returns ``False`` until the
        user presses F1 again.  Use this from the main inference loop:

            with KeyboardController() as kbd:
                while not kbd.stop_requested:
                    if kbd.consume_reset_request():
                        ...  # rebuild your scene state
                    ...
        """
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
        return requested

    # ------------------------------------------------------------------
    # Active-command introspection (read-only)
    # ------------------------------------------------------------------

    def get_active_commands(self) -> dict[str, object]:
        """Snapshot which semantic commands are currently held.

        Returns a dict with six keys:

            ``"left"`` / ``"right"``
                Lists of *semantic* command names (the keys of
                :data:`_LEFT_ARM_KEYMAP` / :data:`_RIGHT_ARM_KEYMAP`,
                e.g. ``"y_pos"``, ``"yaw_neg"``, ``"grip_close"``) for
                that arm, ordered to match :data:`_SEMANTIC_DISPLAY_ORDER`.
            ``"speed_level"``
                1-indexed integer in ``[1, len(speed_levels)]`` -- which
                ``Alt+digit`` slot is currently active.
            ``"speed_scalar"``
                Float -- the actual multiplier applied to translation
                and rotation channels (gripper is unaffected).
            ``"left_gripper"`` / ``"right_gripper"``
                One of ``"open"``, ``"closed"``, ``"neutral"`` -- the
                current per-arm latched gripper state.  In ``"latch"``
                gripper mode this is what is actually emitted on the
                gripper channel every frame; in ``"momentary"`` mode the
                value is bookkeeping-only (the held-key gripper logic is
                what drives the channel) but the snapshot still
                reflects the most recent press.

        The snapshot is consistent across all entries (taken under the
        listener lock); a key press exactly during the call cannot
        split between, say, "old scalar was used for left arm but new
        scalar for right arm".  Intended primarily for live-display
        overlays such as
        ``projects/cosmos_h_surgical/run_keyboard.py``'s on-frame
        command HUD; keep this read-only and side-effect-free.
        """
        with self._lock:
            held = frozenset(self._held_keys)
            level_one_indexed = self._current_speed_level + 1
            scalar = self._current_speed_scalar
            left_grip = self._left_grip_state
            right_grip = self._right_grip_state

        def _active_for_arm(keymap: dict) -> list[str]:
            present = [sem for sem, phys in keymap.items() if phys in held]
            present_set = set(present)
            return [sem for sem in _SEMANTIC_DISPLAY_ORDER if sem in present_set]

        def _grip_label(state: int) -> str:
            return {-1: "closed", 0: "neutral", +1: "open"}[state]

        return {
            "left": _active_for_arm(self.config.left_arm_keys),
            "right": _active_for_arm(self.config.right_arm_keys),
            "speed_level": level_one_indexed,
            "speed_scalar": scalar,
            "left_gripper": _grip_label(left_grip),
            "right_gripper": _grip_label(right_grip),
        }

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
        # Snapshot inside the lock whether this press changed the active
        # speed level / gripper latch so we can print a single-line
        # confirmation outside the lock (avoids holding the listener
        # thread during stdout).
        speed_change: Optional[tuple[int, float]] = None
        gripper_change: Optional[tuple[str, int, int]] = None  # (arm, old, new)
        reset_change: bool = False
        with self._lock:
            if token == self._reset_key:
                # F1: signal the consumer to rebuild scene state.  Also
                # clear both arms' latched gripper state so the rebuilt
                # scene starts at the same neutral baseline as a fresh
                # process (otherwise the freshly-rendered first frame
                # would be overwritten by a stale latched-open / -closed
                # gripper command on the very next AR step).
                self._reset_requested = True
                if self._left_grip_state != 0 or self._right_grip_state != 0:
                    reset_change = True
                self._left_grip_state = 0
                self._right_grip_state = 0
            self._held_keys.add(token)
            # Mirror left / right Alt under the generic ``Key.alt`` alias
            # so the Alt+digit check below is a single membership test.
            # Note we deliberately do NOT mirror ``Key.alt_gr`` (AltGr on
            # EU layouts) so that AltGr+digit -- a normal way to type
            # special characters on QWERTZ -- never triggers a speed swap.
            if token in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
                self._held_keys.add(self._alt_alias)
            # Alt+digit: switch the sticky speed scalar.  Bare digit
            # presses (no Alt) just sit harmlessly in ``_held_keys`` --
            # they are not bound to any action.
            if (
                isinstance(token, str)
                and token in self._speed_digit_keys
                and self._alt_alias in self._held_keys
            ):
                new_level = self._speed_digit_keys[token]
                new_scalar = float(self.config.gains.speed_levels[new_level])
                if new_level != self._current_speed_level:
                    speed_change = (new_level + 1, new_scalar)
                self._current_speed_level = new_level
                self._current_speed_scalar = new_scalar
            # Sticky-gripper update.  We always update the latched state
            # (cheap; lets the user switch ``gripper_mode`` mid-run via
            # the API without losing state); it is only *consumed* by
            # ``_build_action_step`` when ``gripper_mode == "latch"``.
            if token in self._gripper_keys:
                arm, sign = self._gripper_keys[token]
                if arm == "left":
                    if self._left_grip_state != sign:
                        gripper_change = ("LEFT", self._left_grip_state, sign)
                    self._left_grip_state = sign
                else:
                    if self._right_grip_state != sign:
                        gripper_change = ("RIGHT", self._right_grip_state, sign)
                    self._right_grip_state = sign
        if speed_change is not None:
            level_one_indexed, scalar = speed_change
            print(
                f"[keyboard_controller] speed level -> {level_one_indexed} "
                f"({scalar:g}x)",
                flush=True,
            )
        if gripper_change is not None and self.config.gripper_mode == "latch":
            arm, _old, new = gripper_change
            label = {-1: "CLOSED", 0: "neutral", +1: "OPEN"}[new]
            print(
                f"[keyboard_controller] {arm} gripper latched -> {label}",
                flush=True,
            )
        if reset_change and self.config.gripper_mode == "latch":
            print(
                "[keyboard_controller] F1: cleared latched gripper state "
                "(both arms -> neutral)",
                flush=True,
            )
        if token == self._print_key:
            self._print_snapshot()
        return None

    def _on_release(self, key) -> None:
        token = _normalise_key(key)
        if token is None:
            return
        with self._lock:
            self._held_keys.discard(token)
            # Mirror Alt pair: only clear the generic alias when BOTH
            # left and right Alt variants have been released.
            if token in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r):
                if (
                    pynput_keyboard.Key.alt_l not in self._held_keys
                    and pynput_keyboard.Key.alt_r not in self._held_keys
                ):
                    self._held_keys.discard(self._alt_alias)

    # ------------------------------------------------------------------
    # Action sampling (called by the simulator main loop)
    # ------------------------------------------------------------------

    def _build_action_step(self) -> np.ndarray:
        """Compose the un-padded 20-D action for the current key state.

        Returns:
            ``(DVRK_RAW_DIM,)`` ``float32`` vector in normalised space.
            Translation / rotation channels are signed sigma offsets
            (zero when no key is held = "do not move this frame", which
            matches their *relative* training-time semantics).  Gripper
            channels are *absolute* opening targets in sigma units.

            How the gripper channel is filled depends on
            :attr:`KeyboardControllerConfig.gripper_mode`:

            * ``"latch"`` (default): the per-arm latched state set by
              the most recent gripper key press on that arm is emitted
              every frame (``+gripper`` open, ``-gripper`` closed,
              ``0`` neutral / pre-first-press / post-F1).  This is the
              correct behaviour for the absolute gripper channel --
              gripper=0 in the training data is the *dataset-mean*
              opening, not a no-op, so a momentary release would snap
              the jaws back to mid-position mid-trajectory.
            * ``"momentary"``: legacy held-only behaviour; ``+gripper``
              while the open key is physically held, ``-gripper`` while
              close is held, sum to ~0 if both are held, ``0`` (= mean
              opening) when neither is held.
        """
        gains = self.config.gains
        latch_mode = self.config.gripper_mode == "latch"
        with self._lock:
            held = frozenset(self._held_keys)
            # Snapshot the active speed scalar atomically with the held
            # set so a concurrent Alt+digit press cannot split the read
            # ("which scalar applied to which key") across a step.
            speed = self._current_speed_scalar
            # Same atomicity argument applies to the gripper latch
            # state: a press during this snapshot must not split between
            # the held set and the latch.
            left_grip = self._left_grip_state
            right_grip = self._right_grip_state

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
                    action[bind.dim] += bind.coef * gain * speed
                elif bind.kind == "rotation":
                    action[bind.dim] += bind.coef * gains.rotation * speed
                elif bind.kind == "gripper":
                    if latch_mode:
                        # In latch mode we ignore held gripper keys
                        # here -- the latched state below is the single
                        # source of truth for gripper sigma so the user
                        # can press a non-gripper key without snapping
                        # the jaws back to mean.
                        continue
                    # Momentary: absolute target, NOT scaled by the speed
                    # selector.  If both open and close are pressed,
                    # they sum to ~0.
                    action[bind.dim] += bind.coef * gains.gripper

        if latch_mode:
            action[_LEFT_BASE + _GRIPPER_OFFSET] = left_grip * gains.gripper
            action[_RIGHT_BASE + _GRIPPER_OFFSET] = right_grip * gains.gripper

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
            "    Space         : gripper close",
            f"    {grip_open_left}             : gripper open",
            "  RIGHT arm (camera-frame right / PSM1, dims 0-9):",
            "    Up / Down     : y +/-",
            "    Left / Right  : x -/+",
            "    I / K         : z +/-",
            "    U / O         : yaw  -/+    [rot6d dims  4,  6]",
            "    J / L         : pitch +/-   [rot6d dim   5]",
            "    N / M         : roll  -/+   [rot6d dim   8]",
            "    R-Shift       : gripper close",
            "    R-Ctrl        : gripper open",
        ]
        if self.config.gripper_mode == "latch":
            lines += [
                "  Gripper mode  : LATCH  (each gripper key press sticks "
                "until the opposite gripper key on that arm is pressed; F1 "
                "resets both arms to neutral).  Required for the absolute "
                "gripper channel: 0 sigma = dataset-mean opening, NOT a "
                "no-op, so a momentary release would snap the jaws to "
                "mid-position.",
            ]
        else:
            lines += [
                "  Gripper mode  : MOMENTARY  (gripper channel is +/- gain "
                "ONLY while the open / close key is physically held; "
                "release returns to 0 sigma = dataset-mean opening, which "
                "the model will visibly drive towards).",
            ]
        lines += [
            "  Speed levels (sticky; apply to motion + rotation, NOT gripper):",
        ]
        levels = self.config.gains.speed_levels
        default_one_indexed = self.config.gains.default_speed_level
        for i, scalar in enumerate(levels):
            marker = "  <- startup default" if (i + 1) == default_one_indexed else ""
            lines.append(f"    Alt+{i + 1}         : x{scalar:g}{marker}")
        lines += [
            "    (left or right Alt; AltGr is excluded)",
            "  Misc:",
            "    P             : print current 20-D action snapshot",
            "    F1            : reset scene to initial first frame "
            "(consumer-side, e.g. to rebuild an inference KV cache and "
            "shake off accumulated drift; does NOT exit the listener)",
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
