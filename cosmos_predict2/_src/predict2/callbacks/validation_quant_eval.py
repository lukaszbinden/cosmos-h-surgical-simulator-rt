# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""In-training validation quality gate / hook (Option C: async sbatch).

Every ``every_n`` training steps, this callback submits a SLURM job (via
``sbatch``) that runs the full surgical-sim quant evaluation
(FDS + GATC + TCD + a small OOD scenario set) against the latest DCP
checkpoint, then posts results to the same WandB run as the trainer plus a
local snapshot under ``<repo_root>/validation/iter_<iter>/``.

Training itself is never blocked: the callback fires-and-forgets.  If
sbatch is unavailable or queue submission fails, a warning is logged and
training continues unaffected.

Layout produced (per validation):

    <repo_root>/validation/
    ├── metrics_history.csv           # one row per validation, all iters
    └── iter_<iter>/
        ├── metrics.json              # aggregated FDS / GATC / TCD
        ├── quant_eval_results.json   # full per-episode breakdown
        ├── worker.log
        ├── comparison/<dataset>/ep<id>_seed<s>.mp4
        └── ood/<dataset>/ood_scenarios/episode_<id>/{13_depth_push_into,14_depth_pull_away}.mp4
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

import torch
import wandb

from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_predict2._src.imaginaire.utils import distributed, log


# Fixed location relative to /workspace inside the container (CODE_PATH mount)
_TEMPLATE_REL = "train_scripts/_validation_quant_eval_sbatch.sh.template"


class EveryNValidationQuantEval(EveryN):
    """Async validation quant-eval callback.

    Submits an sbatch job at every ``every_n`` iterations that:
      1. Converts the latest DCP checkpoint to a single ``.pt``.
      2. Runs FDS + GATC + TCD on a small validation subset (3 datasets x
         2 episodes x 2 seeds by default).
      3. Generates a small OOD scenario set (depth-only, 1 episode on
         hf_suturebot by default).
      4. Resumes the training run's WandB session and posts ``val/...``
         metrics + a couple of comparison + OOD videos.
      5. Persists everything under ``<repo_root>/validation/iter_<iter>/``.

    All file-system / sbatch operations happen on rank 0 only.

    Args:
        every_n: Frequency (in iterations) at which to submit a validation
            job.  Default 1000.
        repo_root_in_container: Absolute path inside the container that maps
            to the repo root.  Used to resolve the sbatch template and the
            validation output directory.  Defaults to ``/workspace`` (which
            matches the train script's ``CODE_PATH:/workspace`` mount).
        validation_subdir: Directory name under repo_root for validation
            artifacts.  Default ``validation``.
        sam3_checkpoint: Absolute path to the Medical-SAM3 .pt checkpoint
            (required: GATC and TCD need it).  This is host-readable from
            the worker via the ``healthcareeng_holoscan`` mount.
        val_datasets: Subset of validation dataset names (basename match)
            from ``JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS``.
        num_episodes: Episodes per validation dataset.  Default 2.
        num_seeds: Random seeds per episode.  Default 2.
        ood_datasets: Subset of datasets to generate OOD scenarios for.
        ood_episodes: Episodes per OOD dataset.  Default 1.
        ood_depth_only: If True (default), only the 2 depth scenarios are
            generated per episode.
        partition: SLURM partition.  Default 'batch_block1'.
        account: SLURM account.  Default 'healthcareeng_holoscan'.
        time_limit: SLURM walltime, e.g. ``"02:00:00"``.
        container_image: Absolute path to the .sqsh image (host).
        extra_container_mounts: Optional list of extra mounts to append to
            the worker job's ``--container-mounts`` (the workspace and
            healthcareeng_holoscan are mounted automatically).
    """

    def __init__(
        self,
        every_n: int = 1000,
        step_size: int = 1,
        run_at_start: bool = False,
        # paths
        repo_root_in_container: str = "/workspace",
        validation_subdir: str = "validation",
        # SAM3
        sam3_checkpoint: str = (
            "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
            "users/lzbinden/checkpoints/medical_sam3/checkpoint_8_new_best.pt"
        ),
        # validation scope
        val_datasets: Optional[List[str]] = None,
        num_episodes: int = 2,
        num_seeds: int = 2,
        guidance: float = 0.0,
        # OOD scope
        ood_datasets: Optional[List[str]] = None,
        ood_episodes: int = 1,
        ood_depth_only: bool = True,
        # SLURM
        partition: str = "batch_block1",
        account: str = "healthcareeng_holoscan",
        time_limit: str = "02:00:00",
        container_image: str = (
            "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
            "users/lzbinden/images/cosmos-predict-2.5.sqsh"
        ),
        extra_container_mounts: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            every_n=every_n,
            step_size=step_size,
            barrier_after_run=True,
            run_at_start=run_at_start,
        )
        self.repo_root_in_container = repo_root_in_container
        self.validation_subdir = validation_subdir
        self.sam3_checkpoint = sam3_checkpoint
        self.val_datasets = list(val_datasets) if val_datasets else [
            "hf_suturebot", "cosmos_knot_fail_demo", "cosmos_fail_filtered",
        ]
        self.num_episodes = int(num_episodes)
        self.num_seeds = int(num_seeds)
        self.guidance = float(guidance)
        self.ood_datasets = list(ood_datasets) if ood_datasets else ["hf_suturebot"]
        self.ood_episodes = int(ood_episodes)
        self.ood_depth_only = bool(ood_depth_only)
        self.partition = partition
        self.account = account
        self.time_limit = time_limit
        self.container_image = container_image
        self.extra_container_mounts = list(extra_container_mounts) if extra_container_mounts else []
        self.name = self.__class__.__name__

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return

        config_job = self.config.job
        # Trainer writes wandb_id.txt at its on_train_start (wandb_util._write_wandb_id);
        # we read the same path.
        self._wandb_id_file = os.path.join(config_job.path_local, "wandb_id.txt")
        self._checkpoint_root = os.path.join(config_job.path_local, "checkpoints")
        self._validation_root_host = os.path.join(self.repo_root_in_container, self.validation_subdir)
        self._template_path = os.path.join(self.repo_root_in_container, _TEMPLATE_REL)
        self._metrics_history_csv = os.path.join(self._validation_root_host, "metrics_history.csv")

        os.makedirs(self._validation_root_host, exist_ok=True)
        log.info(
            f"[{self.name}] every_n={self.every_n}  "
            f"validation_root={self._validation_root_host}  "
            f"wandb_id_file={self._wandb_id_file}  "
            f"sam3={self.sam3_checkpoint}"
        )

        # Sanity check: the sbatch template file must exist
        if not os.path.isfile(self._template_path):
            log.warning(
                f"[{self.name}] sbatch template not found at {self._template_path}; "
                f"the callback will be a no-op."
            )

        # Sanity check: sbatch must be available somewhere on PATH
        try:
            subprocess.run(["sbatch", "--version"], capture_output=True, check=True, timeout=5)
            log.info(f"[{self.name}] sbatch available; job submission enabled.")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning(
                f"[{self.name}] 'sbatch' is unavailable inside the container ({type(e).__name__}). "
                f"Validation jobs will be skipped (training is unaffected)."
            )

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    @torch.no_grad()
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if not distributed.is_rank0():
            return

        if iteration <= 0:
            log.debug(f"[{self.name}] iteration={iteration} <= 0, skipping.")
            return

        ckpt_dir = self._latest_checkpoint_dir(iteration)
        if ckpt_dir is None:
            log.warning(
                f"[{self.name}] iter={iteration}: no DCP checkpoint dir found under "
                f"{self._checkpoint_root} for this iter. Skipping submission."
            )
            return

        validation_dir = os.path.join(self._validation_root_host, f"iter_{iteration:09d}")
        sbatch_dir = os.path.join(self._validation_root_host, "_sbatch")
        os.makedirs(sbatch_dir, exist_ok=True)
        log_file = os.path.join(sbatch_dir, f"iter_{iteration:09d}.log")
        sbatch_file = os.path.join(sbatch_dir, f"iter_{iteration:09d}.sbatch")

        worker_args = self._build_worker_args(
            iteration=iteration,
            checkpoint_dir=ckpt_dir,
            validation_dir=validation_dir,
        )

        try:
            template = Path(self._template_path).read_text()
        except OSError as exc:
            log.warning(f"[{self.name}] cannot read template {self._template_path}: {exc}")
            return

        filled = (
            template
            .replace("@@JOB_NAME@@", f"cp2.5-val-iter{iteration // 1000:05d}k")
            .replace("@@LOG_FILE@@", log_file)
            .replace("@@PARTITION@@", self.partition)
            .replace("@@ACCOUNT@@", self.account)
            .replace("@@TIME@@", self.time_limit)
            .replace("@@CONTAINER_IMAGE@@", self.container_image)
            .replace("@@CONTAINER_MOUNTS@@", self._build_container_mounts())
            .replace("@@CODE_PATH@@", self.repo_root_in_container)
            .replace("@@WORKER_ARGS@@", worker_args)
        )

        try:
            Path(sbatch_file).write_text(filled)
        except OSError as exc:
            log.warning(f"[{self.name}] cannot write sbatch file {sbatch_file}: {exc}")
            return

        try:
            res = subprocess.run(
                ["sbatch", sbatch_file],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            err = getattr(exc, "stderr", str(exc))
            log.warning(
                f"[{self.name}] iter={iteration}: sbatch submission failed ({type(exc).__name__}): {err}. "
                f"Training continues."
            )
            return

        # Capture the submitted job id (sbatch prints "Submitted batch job <ID>")
        out = (res.stdout or "").strip()
        log.info(f"[{self.name}] iter={iteration}: {out}  (sbatch={sbatch_file})")

        # Best-effort: record the submission in WandB so we can correlate
        # validation runs with the training run timeline.
        try:
            if wandb.run is not None:
                wandb.log(
                    {
                        "val/jobs/submitted": 1.0,
                        "val/jobs/last_iter": iteration,
                    },
                    step=iteration,
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _latest_checkpoint_dir(self, iteration: int) -> Optional[str]:
        """Find the DCP checkpoint dir corresponding to (or just before) iter.

        The trainer writes ``checkpoints/iter_<iter>`` periodically via
        ``checkpoint.save_iter``.  When the validation hook fires at, e.g.,
        iter 2000 with save_iter=200, the iter_000002000 dir is the most
        recent.  If that exact iter dir doesn't exist (e.g., timing
        mismatch), we fall back to the latest iter dir not exceeding the
        callback iter.
        """
        target = os.path.join(self._checkpoint_root, f"iter_{iteration:09d}")
        if os.path.isdir(target) and os.path.isdir(os.path.join(target, "model")):
            return target
        # Fallback: scan for the highest iter_<n> dir <= iteration with a model/ subdir
        try:
            entries = sorted(os.listdir(self._checkpoint_root))
        except OSError:
            return None
        candidate: Optional[str] = None
        for entry in entries:
            if not entry.startswith("iter_"):
                continue
            try:
                n = int(entry[len("iter_"):])
            except ValueError:
                continue
            if n > iteration:
                continue
            full = os.path.join(self._checkpoint_root, entry)
            if os.path.isdir(os.path.join(full, "model")):
                candidate = full
        return candidate

    def _build_container_mounts(self) -> str:
        """Build the comma-separated container mount string for the eval job.

        Mounts are intentionally narrow: only the repo (as /workspace) and
        the broad ``healthcareeng_holoscan`` project mount (which covers
        ``Open-H-lz``, ``Open-H_failures_ood``, the SAM3 checkpoint, and the
        training output dir).
        """
        # CODE_PATH on the host — derive from the repo path that resolves to /workspace.
        # Inside the container the trainer's $PWD is /workspace; on the SLURM submit host
        # we need the host-side absolute path.  We read it from job.path_local's prefix
        # heuristically, but the cleanest source is the trainer's cwd pre-mount.
        # We instead encode the host path via an env var set by the train_scripts/01_train_teacher.sh script.
        code_path_host = os.environ.get(
            "CODE_PATH",
            # Fallback: assume callback runs in the same container as the trainer; the
            # repo is mounted at /workspace and we need the host path. The Cosmos-H-Surgical
            # team's standard layout puts code under
            # /lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/<repo_name>.  The
            # train_scripts/01_train_teacher.sh script exports CODE_PATH for us, so this
            # fallback is only for ad-hoc runs.
            "/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/cosmos-h-surgical-simulator-rt",
        )

        mounts = [
            f"{code_path_host}:{self.repo_root_in_container}",
            "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:"
            "/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan",
            "/lustre/fsw/portfolios/healthcareeng/users/lzbinden:"
            "/lustre/fsw/portfolios/healthcareeng/users/lzbinden",
        ]
        mounts.extend(self.extra_container_mounts)
        return ",".join(mounts)

    def _build_worker_args(
        self,
        iteration: int,
        checkpoint_dir: str,
        validation_dir: str,
    ) -> str:
        """Build the single-line argument string passed to the worker."""
        parts: list[str] = [
            "--experiment", self.config.job.name,
            "--checkpoint-iter", str(iteration),
            "--checkpoint-dir", checkpoint_dir,
            "--validation-dir", validation_dir,
            "--wandb-id-file", self._wandb_id_file,
            "--metrics-history-csv", self._metrics_history_csv,
            "--sam3-checkpoint", self.sam3_checkpoint,
            "--num-episodes", str(self.num_episodes),
            "--num-seeds", str(self.num_seeds),
            "--guidance", str(self.guidance),
            "--ood-episodes", str(self.ood_episodes),
        ]
        if self.ood_depth_only:
            parts.append("--ood-depth-only")
        if self.val_datasets:
            parts += ["--val-datasets", *self.val_datasets]
        if self.ood_datasets:
            parts += ["--ood-datasets", *self.ood_datasets]

        # Quote each so the @@WORKER_ARGS@@ substitution is shell-safe even
        # if a path contains spaces.
        return " ".join(shlex.quote(p) for p in parts)
