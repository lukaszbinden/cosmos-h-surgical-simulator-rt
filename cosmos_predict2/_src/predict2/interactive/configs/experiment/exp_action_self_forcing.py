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

import math

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.distill.utils.config_helper import (
    build_no_s3_run,
    build_no_s3_run_v2,
    deep_update_config_dict,
)
from cosmos_predict2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


def make_experiment(
    name: str,
    data: str,
    model: str = "action_video2world_self_forcing_fsdp",
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
        {"override /net_teacher": "cosmos_v1_2B_action_chunk_conditioned"},
        {"override /net_fake_score": "cosmos_v1_2B_action_chunk_conditioned"},
        {"override /conditioner": conditioner},
        {"override /ckpt_type": "dcp_distill"},
        {"override /optimizer": "fusedadamw"},
        {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
        {"override /checkpoint": "s3"},
        {"override /tokenizer": tokenizer},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(
            group="self_forcing_action",
            name=name,
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            # `fusedadamw` defaults to master_weights=True (FP32 master copy), which is very memory-expensive
            # for 2B-scale nets and can trigger OOM once optimizer state is first materialized.
            master_weights=False,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1.0],
            warm_up_steps=[0],
            cycle_lengths=[400_000],
        ),
        model=dict(
            config=dict(
                adjust_video_noise=True,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                dmd=True,
                grad_clip=True,
                high_sigma_ratio=0.05,
                high_sigma_strategy=str(HighSigmaStrategy.NONE),
                init_student_with_teacher=True,
                intermediate_feature_ids=None,
                loss_scale=0.0,
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,
                max_num_conditional_frames=2,
                max_simulation_steps=1,
                max_simulation_steps_fake=4,
                min_num_conditional_frames=0,
                net=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                net_fake_score=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                net_teacher=dict(
                    action_dim=29,
                    temporal_compression_ratio=4,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=1.0,
                    sac_config=dict(mode="mm_only"),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                optimizer_discriminator_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    master_weights=False,
                ),
                optimizer_fake_score_config=dict(
                    lr=1e-5,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                    # Avoid allocating FP32 master weights for the fake-score optimizer (big memory spike after first step).
                    master_weights=False,
                ),
                rectified_flow_loss_weight_uniform=False,
                resolution="720",
                resize_online=True,
                scaling="rectified_flow",
                sde=dict(
                    p_mean=-0.8,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                sde_D=dict(
                    p_mean=0.0,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                selected_sampling_time=[math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
                sigma_conditional=0.0001,
                sigma_data=1.0,
                state_t=1 + 12 // 4,
                student_update_freq=5,
                tangent_warmup=1,
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                teacher_guidance=0.0,
                text_encoder_class="reason1p1_7B",
                # Enable generating a decoded video during training so the interactive
                # W&B callback can log `train/backward_simulation_video`.
                vis_debug=True,
                vis_debug_every_n=100,
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=False,
                ),
                timestep_shift=5,
            ),
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=True,
        ),
        trainer=dict(
            max_iter=1000,
            logging_iter=50,
            callbacks=dict(
                iter_speed=dict(hit_thres=200),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
                every_n_sample_ema=dict(
                    every_n=25000000000000000000,
                    is_image=False,
                    num_samples=5,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
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

ACTION_GR00T_GR1_SELF_FORCING = make_experiment(
    name="gr1_i4-a",
    data="gr00t_gr1_warmup",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/interactive_warmup/gr1_i4/checkpoints/iter_000002000",
        ),
        model=dict(
            config=dict(
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_gr1_customized_13frame_full_16nodes/checkpoints/iter_000014000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

ACTION_GR00T_G1_SELF_FORCING = make_experiment(
    name="g1",
    data="gr00t_g1_warmup",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            load_path="cosmos_predict2_action_conditioned/interactive_warmup/g1/checkpoints/iter_000020000",
        ),
        model=dict(
            config=dict(
                net=dict(action_dim=43),
                net_fake_score=dict(action_dim=43),
                net_teacher=dict(action_dim=43),
                teacher_load_from=dict(
                    load_path="s3://bucket/cosmos_predict2_action_conditioned/action_conditional/cosmos_predict2p5_2B_action_conditioned_gr00t_g1_gear_wild_merged_customized_13frame_full_16nodes/checkpoints/iter_000038000/model",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
    ),
)

# ============================================================================
# JHU dVRK Mono Self-Forcing (Phase 2 of the SF pipeline for the surgical sim)
# ============================================================================
# Distills a streaming-capable student model from the annealed JHU dVRK teacher
# using the warmup-trained student as init. Output checkpoint is the input to
# the realtime simulator in flashsim-jg.
#
# Recipe rationale (vs upstream gr1, also vs SF debug's SutureBot):
#   - action_dim = 44 in net / net_fake_score / net_teacher  (JHU dVRK MAX_ACTION_DIM)
#   - resolution = "288"                                       (matches teacher's 288x512)
#   - rope_h/w_extrapolation_ratio = 3.0 (inherited from upstream defaults).
#     Critical: our annealed teacher was trained with these exact values
#     (see exp_2B_action_conditioned_rectify_flow_gr00t.py:148-150) -- a RoPE
#     mismatch between student and teacher would systematically shift the
#     student's attention patterns relative to the teacher's targets.
#   - 8 nodes x 8 GPUs x bs=1 = 64 effective batch (vs upstream 16 x 1 = 128).
#     Linear LR scaling: optimizer.lr 1e-7 -> 5e-8;
#                        discriminator/fake_score lr 1e-5 -> 5e-6.
#     Same logic as SF debug used (4 nodes -> 2.5e-8 / 2.5e-6), just for 8 nodes.
#   - max_iter = 1000 inherited from make_experiment defaults (upstream + SF debug).
#   - data = jhu_dvrk_mono_warmup -- reuse the Phase 0 cache (already registered
#     in interactive/configs/data.py); same data the warmup-student was trained on.
#   - checkpoint.load_path = warmup-student DCP (init for self.net + self.net_fake_score).
#   - teacher_load_from.load_path = annealed teacher DCP /model subfolder
#     (init for self.net_teacher).
# ============================================================================
ACTION_JHU_DVRK_MONO_SELF_FORCING = make_experiment(
    name="jhu_dvrk_mono_i4-sf",
    data="jhu_dvrk_mono_warmup",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            # Warmup-student DCP shard from Phase 1 (NOT a .pt file).
            # Used to initialize self.net and (with init_student_with_teacher=True)
            # also self.net_fake_score's matching parameters.
            load_path="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/imaginaire/output/cosmos_predict2_action_conditioned/interactive_warmup/jhu_dvrk_mono_i4_lr3e-5_no_s3_resumable/checkpoints/iter_000020000",
        ),
        # 8 nodes x bs=1 = 64 effective batch (vs upstream 16 x 1 = 128)
        # => linearly scale lr 1e-7 -> 5e-8.
        optimizer=dict(lr=5e-8),
        model=dict(
            config=dict(
                net=dict(action_dim=44),
                net_fake_score=dict(action_dim=44),
                net_teacher=dict(action_dim=44),
                # Same linear scaling for the auxiliary optimizers.
                optimizer_discriminator_config=dict(lr=5e-6),
                optimizer_fake_score_config=dict(lr=5e-6),
                # JHU teacher's training resolution.
                resolution="288",
                # Annealed JHU dVRK teacher DCP. The /model subfolder is the
                # DCP convention (NOT a .pt file). credentials="" because we're
                # loading from local lustre, not S3.
                teacher_load_from=dict(
                    load_path="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/imaginaire/output/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_fine_anneal_4k/checkpoints/iter_000004000/model",
                    credentials="",
                ),
            ),
        ),
    ),
)

cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing",
    node=ACTION_GR00T_GR1_SELF_FORCING,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_g1_self_forcing",
    node=ACTION_GR00T_G1_SELF_FORCING,
)
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_gr00t_gr1_self_forcing_no_s3",
    # Use v2 + local_path=True to skip the S3 lookup at module-import time:
    # ACTION_GR00T_GR1_SELF_FORCING.checkpoint.load_path is a placeholder
    # ("interactive_warmup/gr1_i4/checkpoints/iter_000002000") that is NOT in
    # the local checkpoint DB, so the v1 path's get_checkpoint_path(s3_url)
    # would raise ValueError at import time. Matches SF debug's pattern.
    node=build_no_s3_run_v2(ACTION_GR00T_GR1_SELF_FORCING, local_path=True, resumable=False),
)
# JHU dVRK Mono SF: resumable variant for SLURM array re-runs.
# Same decoupled flags as the JHU warmup registration in exp_action_warmup.py:
#   - resumable=True            -> fixed ..._no_s3_resumable job name so
#                                  path_local is stable across re-runs (Path-1
#                                  resume picks up the latest SF checkpoint).
#   - load_training_state=False -> first cold start loads only the warmup-
#                                  student's `model` weights into self.net (and
#                                  init_student_with_teacher copies into
#                                  self.net_fake_score). Optim / scheduler /
#                                  trainer state are NOT inherited from the
#                                  warmup checkpoint -- the warmup objective
#                                  (MSE on cached teacher latents) is different
#                                  from the SF objective (DMD / GAN / fake-score
#                                  losses on online teacher rollouts), so
#                                  inheriting Adam moments would mis-scale the
#                                  adaptive LR. Re-runs always load all keys
#                                  from path_local (Path-1) regardless.
#   - wandb_mode="online"       -> stream metrics live to wandb.ai (matches
#                                  warmup, matches teacher fine-tune).
# Build_no_s3_run_v2 is used so the JHU-specific overrides (action_dim=44,
# lr=5e-8, resolution, load_path, teacher_load_from, ...) survive the no-S3
# transformation.
cs.store(
    group="experiment",
    package="_global_",
    name="cosmos_predict2p5_2B_action_jhu_dvrk_mono_self_forcing_no_s3_resumable",
    node=build_no_s3_run_v2(
        ACTION_JHU_DVRK_MONO_SELF_FORCING,
        local_path=True,
        resumable=True,
        load_training_state=False,
        wandb_mode="online",
    ),
)
