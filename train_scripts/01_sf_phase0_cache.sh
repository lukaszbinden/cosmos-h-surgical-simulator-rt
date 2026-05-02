#!/bin/bash
# -- Phase 0 of the self-forcing pipeline: generate teacher trajectory cache --
# 1 node x 8 GPUs in parallel. Each GPU rank handles a 1/8 shard of the
# dataset via --start/--end. Already-cached samples are skipped, so requeues
# and array re-runs are safe.
#SBATCH --job-name=sf-jhu-dvrk-phase0-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --account=healthcareeng_holoscan
##SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --time=4:00:00
#SBATCH --time-min=1:30:00
#SBATCH --output=logs/sf-jhu-dvrk-phase0-cache_%A_%a.out
#SBATCH --error=logs/sf-jhu-dvrk-phase0-cache_%A_%a.out
#SBATCH --array=0-1%1
#SBATCH --dependency=singleton
#SBATCH --requeue

# === Phase 0 configuration ===
# Total trajectory cache size. 10000 samples chosen so that even the smallest
# subset (suture_bot_success, ~0.15% of pool) gets ~15 samples; the largest
# (hf_suturebot, ~48%) gets ~4810. Frame-proportional weighting via
# JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS in groot_configs.py.
#
# IMPORTANT: the frame-proportional coverage above is only realised when
# SAMPLE_STRATEGY=random (or uniform). With SAMPLE_STRATEGY=sequential the
# script walks indices [0, TOTAL_SAMPLES) contiguously, which on
# MixedLeRobotDataset hits ONLY the first subset (hf_suturebot) and leaves
# the other 8 subsets at zero samples in the cache. Always use 'random' or
# 'uniform' for multi-subset mixtures.
export TOTAL_SAMPLES=10000

# Sampling strategy for the global index list. 'random' mirrors the warmup
# trainer's DistributedSampler(shuffle=True) and is the recommended default.
# Use the same INDICES_SEED across re-runs / requeues so the same index list
# is rebuilt and skip-existing logic kicks in.
export SAMPLE_STRATEGY=random
export INDICES_SEED=0

# Annealed teacher checkpoint (iter_000004000 from the 4k cosine fine-anneal phase)
export TEACHER_CKPT="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/imaginaire/output/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_fine_anneal_4k/checkpoints/iter_000004000/model_ema_bf16.pt"

# Use the parent (non-anneal) experiment name for the inference config; the
# --ckpt_path override above ensures we load the annealed weights.
export EXPERIMENT="cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss"

# Cache output dir (relative to /workspace = repo root inside container).
# Phase 1 warmup reads from this same path via dataset_jhu_dvrk_mono_warmup.
export SAVE_ROOT="datasets/jhu_dvrk_mono_warmup_4step"

CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/cosmos-h-surgical-simulator-rt"
CONTAINER_IMAGE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/cosmos-predict-2.5.sqsh"

CONTAINER_MOUNTS="$CODE_PATH:/workspace"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/lustre/fsw/portfolios/healthcareeng/users/lzbinden:/lustre/fsw/portfolios/healthcareeng/users/lzbinden"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan"

echo "[jobId=${SLURM_JOB_ID}] Phase 0: Generating teacher trajectory cache for JHU dVRK Mono"
echo "[jobId=${SLURM_JOB_ID}] TEACHER_CKPT.....: ${TEACHER_CKPT}"
echo "[jobId=${SLURM_JOB_ID}] EXPERIMENT.......: ${EXPERIMENT}"
echo "[jobId=${SLURM_JOB_ID}] CACHE PATH.......: ${CODE_PATH}/${SAVE_ROOT}"
echo "[jobId=${SLURM_JOB_ID}] TOTAL SAMPLES....: ${TOTAL_SAMPLES} (split across 8 GPU ranks: ~$((TOTAL_SAMPLES / 8))/rank)"
echo "[jobId=${SLURM_JOB_ID}] SAMPLE STRATEGY..: ${SAMPLE_STRATEGY} (seed=${INDICES_SEED})"
echo "[jobId=${SLURM_JOB_ID}] Array task id....: ${SLURM_ARRAY_TASK_ID:-0}"

srun --export=ALL --ntasks-per-node=8 --gres=gpu:8 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/workspace \
     bash -c '
        # === Per-rank GPU pinning ===
        export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

        # === Activate the venv shipped with cosmos-predict-2.5.sqsh ===
        source .venv/bin/activate

        # === PYTHONPATH for fork-internal modules ===
        export PYTHONPATH=/workspace:${PYTHONPATH:-}

        # === Compute this rank shard of the global index list ===
        # NOTE: --start/--end now slice the GLOBAL ORDERED INDEX LIST built by
        # the inference script per --sample_strategy / --indices_seed (NOT the
        # dataset directly). Each rank rebuilds the same list (deterministic)
        # and processes its slice. Cache files are still named by the actual
        # virtual MixedLeRobotDataset index drawn from that list.
        N_RANKS=8
        SAMPLES_PER_RANK=$(( (TOTAL_SAMPLES + N_RANKS - 1) / N_RANKS ))  # ceil
        START=$(( SLURM_LOCALID * SAMPLES_PER_RANK ))
        END=$(( (SLURM_LOCALID + 1) * SAMPLES_PER_RANK ))
        if [ $END -gt $TOTAL_SAMPLES ]; then END=$TOTAL_SAMPLES; fi

        echo "[Rank $SLURM_LOCALID] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES slot range [$START, $END) of $TOTAL_SAMPLES (strategy=$SAMPLE_STRATEGY, seed=$INDICES_SEED)"

        python cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py \
            --experiment "$EXPERIMENT" \
            --ckpt_path "$TEACHER_CKPT" \
            --save_root "$SAVE_ROOT" \
            --resolution 288,512 \
            --guidance 0 \
            --chunk_size 12 \
            --sample_strategy "$SAMPLE_STRATEGY" \
            --total_samples $TOTAL_SAMPLES \
            --indices_seed $INDICES_SEED \
            --start $START --end $END \
            --query_steps 0,9,18,27,34
     '

echo "[jobId=${SLURM_JOB_ID}] Phase 0 done (or requeue at next array task to resume)."
