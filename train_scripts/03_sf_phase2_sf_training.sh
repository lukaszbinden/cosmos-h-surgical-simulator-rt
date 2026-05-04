#!/bin/bash
# -- Phase 2 of the self-forcing pipeline: distill the streaming student.    --
# -- JHU dVRK Mono variant.                                                  --
#
# Inputs:
#   - Warmup-student DCP (Phase 1 output) at:
#     .../jhu_dvrk_mono_i4_lr3e-5_no_s3_resumable/checkpoints/iter_000020000
#     -> initializes self.net (and via init_student_with_teacher=True also
#        the matching params of self.net_fake_score).
#   - Annealed teacher DCP (Phase 0 of the C-H-S-S pipeline -- the JHU dVRK
#     anneal-4k checkpoint) at:
#     .../...fine_anneal_4k/checkpoints/iter_000004000/model
#     -> initializes self.net_teacher.
#   - Phase 0 trajectory cache at datasets/jhu_dvrk_mono_warmup_4step
#     -> provides (action, image, ode_latents) tuples for SF rollouts.
#
# Output: SF-distilled streaming-capable student DCP, ready to be converted to
# .pt and consumed by flashsim-jg's CosmosDiTNetwork for realtime inference.
#
# Recipe (see exp_action_self_forcing.py::ACTION_JHU_DVRK_MONO_SELF_FORCING for full
# rationale):
#   - 8 nodes x 8 GPUs x bs=1 = 64 effective batch.
#   - lr = 5e-8 (linearly scaled from upstream 1e-7 for 16-node baseline).
#   - max_iter = 1000 (upstream + SF debug default).
#   - action_dim = 44 in net / net_fake_score / net_teacher.
#   - resolution = "288"; rope_h/w_extrapolation_ratio = 3.0 (matches teacher).
#
# Container: imaginaire4:v10.1.7.sqsh (NOT cosmos-predict-2.5.sqsh) -- the
# action_causal_cosmos_v1_2B net used for the student and the fake-score net
# requires NATTEN's multi-dim attention backend, which is only present in
# the imaginaire4 image. cosmos-predict-2.5.sqsh would crash at startup with:
#   ValueError: Could not find a compatible Multi-Dimensional Attention
#   backend for this use case / device.
# Mirrors SF debug's sf_training.sh runtime setup exactly.
#SBATCH --job-name=sf-jhu-dvrk-sf
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --account=healthcareeng_holoscan
##SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --time=4:00:00
#SBATCH --time-min=2:00:00
#SBATCH --output=logs/sf-jhu-dvrk-sf_%A_%a.out
#SBATCH --error=logs/sf-jhu-dvrk-sf_%A_%a.out
#SBATCH --array=0-1%1
#SBATCH --dependency=singleton
#SBATCH --requeue

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# Archive repository only on first array task to save space
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
  # SLURM-safe repo root detection using pyproject.toml as marker
  _d="${SLURM_SUBMIT_DIR:-$PWD}"; while [[ "$_d" != "/" && ! -f "$_d/pyproject.toml" ]]; do _d="$(dirname "$_d")"; done
  REPO_ROOT="$_d"; [[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: Cannot find repo root"; exit 1; }
  REPO_DEST="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/imaginaire/output/repo/sf-jhu-dvrk-sf/run_${TIMESTAMP}_${SLURM_JOB_NAME:-unknown}"
  mkdir -p "$REPO_DEST" && tar -czf "$REPO_DEST/repo.tar.gz" -C "$REPO_ROOT" \
    --exclude='*.log' --exclude='*.out' --exclude='*.pt' --exclude='*.pth' --exclude='*.bin' --exclude='*.onnx' \
    --exclude='*.npy' --exclude='*.npz' --exclude='*.parquet' --exclude='*.h5' --exclude='*.safetensors' \
    --exclude='*.mp4' --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.gif' --exclude='*.sqsh' \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='logs' --exclude='uv.lock' \
    --exclude='.venv' --exclude='venv' --exclude='.env' --exclude='*.egg-info' --exclude='.eggs' \
    --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' --exclude='dist' --exclude='build' --exclude='datasets' . \
    && cat > "$REPO_DEST/README.txt" <<EOF
SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}
SLURM_JOB_NAME....: ${SLURM_JOB_NAME}
TIMESTAMP.........: ${TIMESTAMP}
REPO_ROOT.........: ${REPO_ROOT}
LOG_FILES.........: ${REPO_ROOT}/logs/sf-jhu-dvrk-sf_${SLURM_ARRAY_JOB_ID}_*.out
EOF
  echo "[jobId=${SLURM_JOB_ID}] Repo archived to: $REPO_DEST/repo.tar.gz"
else
  echo "[jobId=${SLURM_JOB_ID}] Skipping repo archive (only done on array task 0)"
fi

# === IMAGINAIRE output / cache layout ===
# Same pattern as the warmup launcher: SF training writes checkpoints to the
# standard fsw lustre path so they're discoverable for downstream conversion
# (DCP -> model_ema_bf16.pt for flashsim-jg). Mounted as /imaginaire_output
# inside the container; IMAGINAIRE_OUTPUT_ROOT then redirects writes there.
OUTPUT_DIR="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/imaginaire/output"
CACHE_DIR="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/imaginaire/cache"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
echo "[jobId=${SLURM_JOB_ID}] imaginaire_output: ${OUTPUT_DIR}"
echo "[jobId=${SLURM_JOB_ID}] imaginaire_cache.: ${CACHE_DIR}"
echo "[jobId=${SLURM_JOB_ID}] Array Task ID....: ${SLURM_ARRAY_TASK_ID}"

# Compute distributed training variables (exported to container via --export=ALL)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Nodelist=$SLURM_JOB_NODELIST"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"

CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/cosmos-h-surgical-simulator-rt"
CONTAINER_IMAGE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/imaginaire4:v10.1.7.sqsh"

CONTAINER_MOUNTS="${OUTPUT_DIR}:/imaginaire_output"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},$CODE_PATH:/workspace"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${CACHE_DIR}:/imaginaire_cache"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan"

srun --export=ALL \
     --container-image="$CONTAINER_IMAGE" --container-name=container \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/workspace \
     bash -c '
        # === DISTRIBUTED TRAINING SETUP ===
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        echo "[Rank $RANK / LOCAL_RANK $LOCAL_RANK] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

        # === PYTHON SETUP ===
        # imaginaire4 image: use container Python directly (no .venv activation).
        # Add fork-internal packages to PYTHONPATH (cosmos_oss, cosmos_cuda, cosmos_gradio).
        export PANDAS_NO_EXTENSION_ARRAY=True
        export PYTHONPATH=/workspace:/workspace/packages/cosmos-oss:/workspace/packages/cosmos-cuda:/workspace/packages/cosmos-gradio:${PYTHONPATH:-}

        # Install missing packages on LOCAL_RANK 0 only to avoid race conditions.
        # python -m pip avoids the broken /root/.local/bin/pip shebang in the container.
        if [ "$LOCAL_RANK" = "0" ]; then
            python -m pip install tyro albumentations --quiet
        fi
        sleep 10

        # Sanity check NATTEN is available (the SF student / fake-score nets need it):
        if [ "$LOCAL_RANK" = "0" ]; then
            echo "[Rank $RANK] Python: $(which python)"
            echo "[Rank $RANK] PYTHONPATH: $PYTHONPATH"
            python -c "import torch; print(\"[Rank $RANK] CUDA available:\", torch.cuda.is_available(), \"device count:\", torch.cuda.device_count())"
            python -c "import natten; print(\"[Rank $RANK] NATTEN version:\", natten.__version__)" 2>&1 || echo "[Rank $RANK] NATTEN import failed -- SF training will crash"
            python -c "import natten; from natten.functional import na2d; print(\"[Rank $RANK] NATTEN na2d available\")" 2>&1 || echo "[Rank $RANK] NATTEN na2d not available"
        fi

        # === TRAINING ENVIRONMENT ===
        ulimit -c 0
        export TORCH_NCCL_ENABLE_MONITORING=0
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export IMAGINAIRE_OUTPUT_ROOT=/imaginaire_output
        export IMAGINAIRE_CACHE_DIR=/imaginaire_cache
        export TORCH_HOME=/imaginaire_cache
        export WANDB_CACHE_DIR=/imaginaire_cache
        export WANDB_DATA_DIR=/imaginaire_cache
        export ENABLE_ONELOGGER=TRUE

        # === RUN TRAINING ===
        # python -m scripts.train (NOT torchrun): the trainer reads
        # RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT from env,
        # which the per-rank srun layout (--ntasks-per-node=8) sets correctly.
        # config_distill.py wires up the distillation-flavored ckpt_type
        # ("dcp_distill") and callbacks needed for the SF training loop.
        python -m scripts.train \
            --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
            -- \
            experiment=cosmos_predict2p5_2B_action_jhu_dvrk_mono_self_forcing_no_s3_resumable \
            checkpoint.save_iter=200
     '
