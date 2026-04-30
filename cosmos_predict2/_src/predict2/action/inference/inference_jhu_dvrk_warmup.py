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
JHU dVRK Mono variant of the GR00T warmup inference script (Phase 0 of the
self-forcing pipeline). Generates teacher trajectory caches over the 9-dataset
JHU dVRK mono training mixture (frame-proportional weighted via
``JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS``) instead of a single dataset.

Output layout matches the upstream warmup cache convention so it is consumed
unchanged by ``ActionDatasetSFWarmup`` in Phase 1:

    <save_root>/
        latents/<idx>.pt    # teacher denoising-trajectory latents at query_steps
        images/<idx>.png    # first conditioning frame
        actions/<idx>.json  # the 12-step action chunk (44D after MixedLeRobotDataset zero-pad)
        videos/<idx>.mp4    # the 13-frame ground-truth video clip

Already-existing samples in ``<save_root>`` are skipped, so the script can be
re-run / requeued safely (each rank only generates the ones it owns and that
are not yet on disk).

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \\
        cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss \\
        --ckpt_path /lustre/.../iter_000004000/model_ema_bf16.pt \\
        --save_root datasets/jhu_dvrk_mono_warmup_4step \\
        --resolution 288,512 --guidance 0 --chunk_size 12 \\
        --start 0 --end 10000 --query_steps 0,9,18,27,34
"""

import argparse
import json
import os

import mediapy
import numpy as np
import torch
import tqdm
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import MixedLeRobotDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS,
    MAX_ACTION_DIM,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the JHU dVRK warmup cache generator."""
    parser = argparse.ArgumentParser(description="JHU dVRK Mono Phase 0 warmup-cache generator")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--chunk_size", type=int, default=12, help="Action chunk size (must match teacher training)")
    parser.add_argument("--guidance", type=float, default=0.0, help="Classifier-free guidance scale (0 = no CFG)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the teacher checkpoint (.pt file or DCP dir). If empty, falls back to the experiment's load_path.",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--resolution",
        type=str,
        default="288,512",
        help="Resolution of the rendered video, format H,W. Default 288,512 matches the JHU dVRK teacher training.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="datasets/jhu_dvrk_mono_warmup_4step",
        help="Output directory (relative to repo root). Sub-dirs latents/ images/ actions/ videos/ are created.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive) for this rank's shard")
    parser.add_argument("--end", type=int, default=10000, help="End index (exclusive) for this rank's shard")
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (warmup uses 1 = single image conditioning).",
    )
    parser.add_argument(
        "--query_steps",
        type=lambda x: [int(i) for i in x.split(",")],
        default=[0, 9, 18, 27, 34],
        help="Denoising-step indices at which to snapshot the teacher's latent trajectory.",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (default 1 = no CP). Set to 8 if launching on 8 GPUs jointly.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip indices whose cache files already exist on disk. Default True (safe re-run / requeue).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Force regenerate even if cache files already exist.",
    )
    return parser.parse_args()


def _is_already_cached(save_root: str, idx: int) -> bool:
    """Return True iff all 4 cache artefacts for ``idx`` already exist on disk."""
    return (
        os.path.exists(os.path.join(save_root, "latents", f"{idx}.pt"))
        and os.path.exists(os.path.join(save_root, "images", f"{idx}.png"))
        and os.path.exists(os.path.join(save_root, "actions", f"{idx}.json"))
        and os.path.exists(os.path.join(save_root, "videos", f"{idx}.mp4"))
    )


def main():
    torch.enable_grad(False)
    args = parse_arguments()

    # Reproducibility for the dataset's internal sample shuffling.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build the JHU dVRK Mono training mixture (9 datasets, frame-proportional
    # weighting via mix_ratio = total_frames per subset). Using data_split="train"
    # ensures we cache from the same partition the teacher saw during fine-tuning,
    # honoring per-spec test_split_ratio_override (0%-2%) and the ood spec's
    # data_split_override="full".
    dataset = MixedLeRobotDataset(
        dataset_specs=JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS,
        num_frames=13,
        data_split="train",
        max_action_dim=MAX_ACTION_DIM,
        downscaled_res=False,
        test_split_ratio=0.02,
    )
    logger.info(
        f"MixedLeRobotDataset built: virtual size = {len(dataset)}, "
        f"max_action_dim = {MAX_ACTION_DIM}, "
        f"requesting indices [{args.start}, {args.end})"
    )

    # Initialize the inference handler with context parallel support
    video2world_cli = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    save_root = args.save_root
    os.makedirs(os.path.join(save_root, "latents"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "actions"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "videos"), exist_ok=True)

    n_skipped = 0
    n_generated = 0
    indices = list(range(args.start, min(args.end, len(dataset))))
    logger.info(f"Processing {len(indices)} indices on this rank")

    for idx in tqdm.tqdm(indices, desc=f"Phase 0 cache [{args.start},{args.end})"):
        if args.skip_existing and _is_already_cached(save_root, idx):
            n_skipped += 1
            continue

        data = dataset[idx]
        # data["video"] is shape (C, T, H, W) in uint8. Match the upstream warmup
        # convention: feed the FIRST frame as the conditioning image; the action
        # chunk and the 13-frame ground-truth video are saved alongside.
        img_np_array = data["video"][:, 0, :, :].permute(1, 2, 0).cpu().numpy()
        video_np_array = data["video"].permute(1, 2, 3, 0).cpu().numpy()
        action = data["action"].cpu().numpy()

        next_img_array, video_clamped, latents_to_save = video2world_cli.step_inference_with_latents(
            img_array=img_np_array,
            action=action,
            guidance=args.guidance,
            seed=args.seed,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            query_steps=args.query_steps,
        )

        for k in latents_to_save:
            latents_to_save[k] = latents_to_save[k].squeeze(0).cpu()

        torch.save(latents_to_save, os.path.join(save_root, "latents", f"{idx}.pt"))
        mediapy.write_image(os.path.join(save_root, "images", f"{idx}.png"), img_np_array)
        mediapy.write_video(os.path.join(save_root, "videos", f"{idx}.mp4"), video_np_array)
        with open(os.path.join(save_root, "actions", f"{idx}.json"), "w") as f:
            json.dump(action.tolist(), f, indent=4)
        n_generated += 1

    logger.info(
        f"Phase 0 cache shard [{args.start},{args.end}) complete: "
        f"generated {n_generated}, skipped (already cached) {n_skipped}"
    )


if __name__ == "__main__":
    main()
