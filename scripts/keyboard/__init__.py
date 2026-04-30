# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyboard control utilities for the Cosmos-H-Surgical-Simulator.

Marker file that makes ``scripts.keyboard`` a regular Python package so that
imports like ``from scripts.keyboard.keyboard_controller import ...`` resolve
unambiguously regardless of how the parent script is launched.

The other keyboard files in this directory are intentionally NOT re-exported
here -- import them directly via their fully-qualified path so callers see the
exact module they depend on.
"""
