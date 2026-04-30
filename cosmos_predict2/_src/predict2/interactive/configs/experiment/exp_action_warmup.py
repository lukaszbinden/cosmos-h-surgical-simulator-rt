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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.distill.utils.config_helper import (
    build_no_s3_run,
    build_no_s3_run_v2,
    deep_update_config_dict,
)


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_warmup_fsdp",
    net: str = "action_causal_cosmos_v1_2B",
    conditioner: str = "video_action_conditioner",
    tokenizer: str = "wan2pt1_tokenizer",
    overrides: dict | None = None,
) -> LazyDict:
    defaults = [
        {"override /data_train": data},
        {"override /data_val": data},
        {"override /model": model},
        {"override /net": net},
        {"override /conditioner": conditioner},
        {"override /tokenizer": tokenizer},
        {"override /ckpt_type": "dcp"},
        {"override /checkpoint": "s3"},
        {"override /callbacks": ["basic_warmup", "wandb_warmup", "cluster_speed"]},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_warmup",
            name=name,
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(enabled=True),
            load_from_object_store=dict(enabled=True),
            load_training_state=False,
            strict_resume=True,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=0.1,
            betas=[0.9, 0.999],
            master_weights=False,
        ),
        scheduler=dict(
            warm_up_steps=[0],
            f_min=[1.0],
            f_max=[1.0],
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=29,
                    num_action_per_chunk=12,
                    timestep_scale=0.001,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                resolution=720,
            ),
        ),
        trainer=dict(
            max_iter=20000,
            logging_iter=20,
            callbacks=dict(
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                manual_gc=dict(
                    every_n=200,
                ),
                every_n_sample_reg=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
                every_n_sample_ema=dict(
                    every_n=5000000000000000000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            pin_memory=False,
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update_config_dict(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


####################################
# Create and register experiments #
####################################

ACTION_GR00T_WARMUP_GR1 = make_experiment(
    name="gr1_i4_lr3e-5",
    data="gr00t_gr1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000",
        ),
    ),
)

ACTION_GR00T_WARMUP_G1 = make_experiment(
    name="g1",
    data="gr00t_g1_warmup",
    overrides=dict(
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_g1_gear_wild_merged_customized_13frame_full_16nodes/checkpoints/iter_000038000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=43),
            ),
        ),
    ),
)

# ============================================================================
# JHU dVRK Mono Warmup (Phase 1 of the SF pipeline for the surgical simulator)
# ============================================================================
# Loads the annealed JHU dVRK Mono teacher (iter_000004000 from the 4k cosine
# fine-anneal phase, see exp_2B_action_conditioned_rectify_flow_gr00t.py) and
# trains the student on the Phase 0 trajectory cache produced by
# ``inference_jhu_dvrk_warmup.py`` over the 9-dataset JHU dVRK mixture.
#
# Recipe (vs. upstream gr1):
#   - action_dim = 44   (matches the JHU dVRK teacher's MAX_ACTION_DIM padding)
#   - resolution = 288  (the teacher trained at 288x512 W,H -- registry value)
#   - 8 nodes x 8 GPUs x batch_size=8 = effective batch 512 (matches upstream's
#     16 nodes x bs=4 = 512), so we keep upstream's lr=3e-5 unchanged.
#   - max_iter = 20000  (per the SF instruction RTF).
# ============================================================================
ACTION_JHU_DVRK_MONO_WARMUP = make_experiment(
    name="jhu_dvrk_mono_i4_lr3e-5",
    data="jhu_dvrk_mono_warmup",
    overrides=dict(
        checkpoint=dict(
            # Annealed JHU dVRK Mono teacher (DCP shard, NOT a .pt). The DCP
            # contains both net.* and net_ema.*; load_ema_to_reg in the warmup
            # model loads net_ema.* into the student's net.*.
            load_path="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/imaginaire/output/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_fine_anneal_4k/checkpoints/iter_000004000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=44),
                resolution="288",  # 288x512 (W,H) per EMBODIMENT_REGISTRY['jhu_dvrk_mono']
            ),
        ),
        # 8 nodes x bs=8 = 512 effective batch ≡ upstream 16 nodes x bs=4
        # ⇒ same per-sample LR as upstream's tested 3e-5.
        optimizer=dict(lr=3e-5),
        dataloader_train=dict(
            batch_size=8,
        ),
    ),
)

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py -- experiment=cosmos_predict2p5_2B_action_gr00t_gr1_warmup
"""

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup",
    node=ACTION_GR00T_WARMUP_GR1,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_g1_warmup",
    node=ACTION_GR00T_WARMUP_G1,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_warmup_no_s3",
    node=build_no_s3_run(ACTION_GR00T_WARMUP_GR1),
)
# JHU dVRK Mono warmup: resumable variant for SLURM array re-runs (fixed job
# name across restarts so checkpoints land in / load from the same dir). Uses
# build_no_s3_run_v2 so the JHU-specific overrides (action_dim=44, batch_size=8,
# lr=3e-5, load_path, resolution) survive the no-S3 transformation.
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_jhu_dvrk_mono_warmup_no_s3_resumable",
    node=build_no_s3_run_v2(ACTION_JHU_DVRK_MONO_WARMUP, local_path=True, resumable=True),
)
