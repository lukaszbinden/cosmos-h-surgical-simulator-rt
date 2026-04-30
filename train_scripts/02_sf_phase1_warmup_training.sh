#!/bin/bash
# -- Phase 1 of the self-forcing pipeline: warmup the student model on the --
# -- Phase 0 teacher trajectory cache. JHU dVRK Mono variant.              --
#
# Reads cached teacher trajectories from datasets/jhu_dvrk_mono_warmup_4step
# (produced by 01_sf_phase0_cache.sh). Trains the student on (action, image,
# latents) tuples to imitate the teacher's denoising trajectory at the
# query_steps. Output: an iter_<N> warmup-student DCP that becomes the input
# to Phase 2 self-forcing distillation (03_sf_phase2_sf_training.sh).
#
# Recipe: 8 nodes x 8 GPUs x bs=8 = effective batch 512 (== upstream 16 x 4),
# lr=3e-5, max_iter=20000, constant LR. See exp_action_warmup.py for details.
#SBATCH --job-name=sf-jhu-dvrk-warmup
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --account=healthcareeng_holoscan
##SBATCH --partition=batch_block1,batch_block3,batch_block4
#SBATCH --time=4:00:00
#SBATCH --time-min=2:00:00
#SBATCH --output=logs/sf-jhu-dvrk-warmup_%A_%a.out
#SBATCH --error=logs/sf-jhu-dvrk-warmup_%A_%a.out
#SBATCH --array=0-9%1
#SBATCH --dependency=singleton
#SBATCH --requeue

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# Archive repository only on first array task to save space
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
  # SLURM-safe repo root detection using pyproject.toml as marker
  _d="${SLURM_SUBMIT_DIR:-$PWD}"; while [[ "$_d" != "/" && ! -f "$_d/pyproject.toml" ]]; do _d="$(dirname "$_d")"; done
  REPO_ROOT="$_d"; [[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "ERROR: Cannot find repo root"; exit 1; }
  REPO_DEST="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/imaginaire/output/repo/sf-jhu-dvrk-warmup/run_${TIMESTAMP}_${SLURM_JOB_NAME:-unknown}"
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
LOG_FILES.........: ${REPO_ROOT}/logs/sf-jhu-dvrk-warmup_${SLURM_ARRAY_JOB_ID}_*.out
EOF
  echo "[jobId=${SLURM_JOB_ID}] Repo archived to: $REPO_DEST/repo.tar.gz"
else
  echo "[jobId=${SLURM_JOB_ID}] Skipping repo archive (only done on array task 0)"
fi

# === Distributed training environment ===
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
nodes_array=("${nodes[@]}")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

echo "[jobId=${SLURM_JOB_ID}] MASTER_ADDR=$MASTER_ADDR  Head node IP=$head_node_ip"
echo "[jobId=${SLURM_JOB_ID}] All nodes:    ${nodes_array[@]}"
echo "[jobId=${SLURM_JOB_ID}] Array task id: ${SLURM_ARRAY_TASK_ID:-0}"

# === Container & mounts ===
CODE_PATH="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/cosmos-h-surgical-simulator-rt"
CONTAINER_IMAGE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/cosmos-predict-2.5.sqsh"

CONTAINER_MOUNTS="$CODE_PATH:/workspace"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/lustre/fsw/portfolios/healthcareeng/users/lzbinden:/lustre/fsw/portfolios/healthcareeng/users/lzbinden"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan"

srun --export=ALL --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --container-workdir=/workspace \
     bash -c '
        echo "MASTER_ADDR=$MASTER_ADDR"
        export NCCL_DEBUG=INFO
        CURRENT_RANK=${SLURM_NODEID:-"0"}
        n_node=${SLURM_JOB_NUM_NODES:-1}
        echo "[jobId=$SLURM_JOB_ID] Array=$SLURM_ARRAY_TASK_ID | Worker_list=$worker_list | Node rank=$CURRENT_RANK of $n_node"

        cd /workspace
        source .venv/bin/activate

        seed=$((1234 + $SLURM_ARRAY_TASK_ID * $n_node * 8))

        # NOTE: unlike 01_train_teacher.sh and 02_train_fine_anneal.sh (which use the
        # action_conditioned config with a mock IterativeJointDataLoader.dataloaders dict),
        # the interactive warmup config replaces data_train wholesale with a single
        # L(DataLoader) (no dataloaders dict). So no ~dataloader_train.dataloaders here
        # -- matches the SF debug warmup_training.sh pattern.
        torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK -m \
            scripts.train \
              --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
              -- \
              experiment="cosmos_predict2p5_2B_action_jhu_dvrk_mono_warmup_no_s3_resumable" \
              checkpoint.save_iter=200
     '
