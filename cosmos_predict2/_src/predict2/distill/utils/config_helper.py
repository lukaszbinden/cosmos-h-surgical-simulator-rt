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
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path


def build_no_s3_run(
    job: dict,
    local_path: bool = False,
    resumable: bool = False,
    load_training_state: bool | None = None,
    wandb_mode: str = "offline",
) -> dict:
    """
    Make a copy of the input config that doesn't require S3 for checkpointing
    and I/O in the callbacks.

    Args:
        job: Source experiment dict.
        local_path: If True, use ``job["checkpoint"]["load_path"]`` verbatim
            instead of resolving it as an S3 URI.
        resumable: If True, use a fixed ``..._no_s3_resumable`` job name (so
            ``path_local`` is stable across SLURM re-runs and the trainer's
            ``latest_checkpoint.txt`` resume logic kicks in). If False, the
            job name gets a timestamp suffix.
        load_training_state: Override the value written into the ``checkpoint``
            override block. If ``None`` (default), defaults to ``resumable``
            for backwards compatibility. Set explicitly to decouple the two
            concepts -- e.g. ``resumable=True, load_training_state=False`` to
            keep fixed path_local but NOT inherit the teacher checkpoint's
            optimizer/scheduler/trainer state on the first run (cleaner Adam
            warmup for objectives that differ from the teacher's loss).
        wandb_mode: WandB sync mode. Default ``"offline"`` for back-compat
            (the legacy "no_s3" semantics bundled WandB cloud sync with the
            checkpoint S3 backend, which is wrong: WandB cloud uses its own
            HTTPS API independent of our S3 setup). Set to ``"online"`` to
            stream metrics live to wandb.ai (overrides ``wandb_util.py``'s
            explicit ``mode=`` kwarg, which ignores ``WANDB_MODE`` env var).
    """
    # If local_path is True, use the local path as the load path
    if local_path:
        load_path = job["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)
    defaults = job.get("defaults", [])

    # For resumable runs, use a fixed name without timestamp so checkpoints
    # are saved to and loaded from the same directory across job restarts.
    if resumable:
        job_name = f"{job['job']['name']}_no_s3_resumable"
    else:
        job_name = f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Default load_training_state to resumable's value (back-compat).
    if load_training_state is None:
        load_training_state = resumable

    no_s3_run = dict(
        defaults=defaults + ["_self_"] if "_self_" not in defaults else defaults,
        job=dict(
            name=job_name,
            wandb_mode=wandb_mode,
        ),
        checkpoint=dict(
            save_to_object_store=dict(enabled=False, credentials=""),
            load_from_object_store=dict(enabled=False),
            load_path=load_path,
            # On the very first run (no path_local checkpoint yet), this gates
            # whether the trainer pulls just ``model`` (False) or all four
            # KEYS_TO_SAVE = model + optim + scheduler + trainer (True) from
            # the load_path. On subsequent SLURM re-runs the trainer's Path-1
            # resume logic (cosmos_predict2/_src/predict2/checkpointer/dcp.py
            # ::keys_to_resume_during_load) ALWAYS loads all four keys from
            # path_local/latest_checkpoint.txt -- so this flag does NOT affect
            # re-run behavior, only the first cold start.
            load_training_state=load_training_state,
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False),
                every_n_sample_ema=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    )
    return no_s3_run


def build_no_s3_run_v2(
    job: dict,
    local_path: bool = False,
    resumable: bool = False,
    load_training_state: bool | None = None,
    wandb_mode: str = "offline",
) -> dict:
    """
    Make a copy of the input config that doesn't require S3 for checkpointing
    and I/O in the callbacks.

    This function creates a deep copy of the original job config and applies
    no-S3 specific overrides on top of it, preserving all original config
    values (like model, optimizer, dataloader_train, etc.).

    This is the fixed version of ``build_no_s3_run`` that preserves all config
    overrides set on the original LazyDict by the experiment's ``make_experiment``
    helper. Required for warmup / SF experiments that need their per-experiment
    overrides (action_dim, batch_size, lr, load_path, ...) to survive the no-S3
    transformation.

    Args:
        job: Source experiment LazyDict / dict.
        local_path: If True, use ``job["checkpoint"]["load_path"]`` verbatim
            instead of resolving via ``get_checkpoint_path`` (S3 lookup).
        resumable: If True, use a fixed ``..._no_s3_resumable`` job name so
            ``path_local`` is stable across SLURM re-runs.
        load_training_state: Override the value written into the ``checkpoint``
            override block. If ``None`` (default), defaults to ``resumable``
            for back-compat. Set explicitly to decouple. See
            ``build_no_s3_run`` for full explanation of the trainer's Path-1
            vs Path-2 resume logic and why this flag only matters on the first
            cold start (not on SLURM re-runs).
        wandb_mode: WandB sync mode. Default ``"offline"`` for back-compat.
            See ``build_no_s3_run`` for the rationale on why this is decoupled
            from the checkpoint S3 backend setting.
    """
    from copy import deepcopy

    from omegaconf import OmegaConf

    # Convert OmegaConf/DictConfig to regular dict for deep copying
    if hasattr(job, "items"):
        # Handle both regular dicts and OmegaConf objects
        try:
            job_dict = OmegaConf.to_container(job, resolve=False)
        except Exception:
            job_dict = dict(job)
    else:
        job_dict = dict(job)

    # Start with a deep copy of the original job config to preserve all settings
    no_s3_run = deepcopy(job_dict)

    # If local_path is True, use the local path as the load path
    if local_path:
        load_path = job_dict["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job_dict['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)

    # For resumable runs, use a fixed name without timestamp so checkpoints
    # are saved to and loaded from the same directory across job restarts.
    if resumable:
        job_name = f"{job_dict['job']['name']}_no_s3_resumable"
    else:
        job_name = f"{job_dict['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    # Default load_training_state to resumable's value (back-compat).
    if load_training_state is None:
        load_training_state = resumable

    # Define the no-S3 specific overrides
    no_s3_overrides = dict(
        job=dict(
            name=job_name,
            wandb_mode=wandb_mode,
        ),
        checkpoint=dict(
            save_to_object_store=dict(enabled=False, credentials=""),
            load_from_object_store=dict(enabled=False),
            load_path=load_path,
            # See build_no_s3_run for explanation: only affects the very first
            # cold start (Path-2 resume); SLURM re-runs always load all keys
            # from path_local/latest_checkpoint.txt (Path-1 resume).
            load_training_state=load_training_state,
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False),
                every_n_sample_ema=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    )

    # Apply the no-S3 overrides on top of the original config
    deep_update_config_dict(no_s3_run, no_s3_overrides)

    # Ensure defaults has "_self_" if not already present
    defaults = no_s3_run.get("defaults", [])
    if "_self_" not in defaults:
        no_s3_run["defaults"] = defaults + ["_self_"]

    return no_s3_run


def deep_update_config_dict(dst: dict, src: dict) -> dict:
    """
    Updates nested dictionaries in the config dictionary (dst) with the values in src dictionary.
    Standard update in hydra only goes one level deep. This function goes arbitrarily deep.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update_config_dict(dst[k], v)
        else:
            dst[k] = v
    return dst
