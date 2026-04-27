#!/usr/bin/env bash
# Run quantitative evaluation on the 4 CMR Versius datasets ONLY.
# Uses the legacy CMR-only mode (no --test_episodes_json).
#
# Datasets evaluated (hardcoded in DATASET_CONFIGS):
#   - prostatectomy_360p
#   - inguinal_hernia_360p
#   - hysterectomy_360p
#   - cholecystectomy_360p

# ── Configuration ─────────────────────────────────────────────────────────────
EXPERIMENT="cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss-scratch"
CKPTS="iter_000004000"
CKPTS="iter_000006000"
CKPTS="iter_000008000 iter_000010000 iter_000012000"
CKPTS="iter_000014000 iter_000016000 iter_000018000 iter_000020000 iter_000022000"
CKPTS="iter_000024000 iter_000026000 iter_000028000 iter_000030000"
CKPTS="iter_000032000 iter_000034000 iter_000036000 iter_000038000"
NUM_EPISODES=2
NUM_SEEDS=2

# Video output (set to "true" to save generated + side-by-side comparison videos)
SAVE_VIDEOS="true"

# Paths
CKPT_BASE_DIR="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/imaginaire/output/cosmos_predict2_action_conditioned/official_runs_vid2vid/${EXPERIMENT}/checkpoints"
SAM3_CKPT="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/checkpoints/medical_sam3/checkpoint_8_new_best.pt"

# Slurm / container
PARTITION="batch_block1"
ACCOUNT="healthcareeng_holoscan"
TIME="04:00:00"
CONTAINER_IMAGE="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/lzbinden/images/cosmos-predict-2.5.sqsh"
WORKSPACE="/lustre/fsw/portfolios/healthcareeng/users/lzbinden/git/scratch-open-h-cp2.5-all"

# ── Launch ────────────────────────────────────────────────────────────────────
srun --job-name=cp2.5-eval-cmr \
  --nodes=1 --ntasks=1 --gres=gpu:1 \
  --partition="${PARTITION}" \
  --account="${ACCOUNT}" \
  --time="${TIME}" \
  --export=ALL \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${WORKSPACE}:/workspace,\
/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan:/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan,\
/lustre/fsw/portfolios/healthcareeng/users/lzbinden:/lustre/fsw/portfolios/healthcareeng/users/lzbinden,\
/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/datasets/Open-H/cmr-surgical-60hz-fixed:/CMR_Versius" \
  --container-workdir=/workspace \
  bash -lc '
set -euo pipefail
source .venv/bin/activate
mkdir -p logs

EXPERIMENT="'"${EXPERIMENT}"'"
CKPT_BASE_DIR="'"${CKPT_BASE_DIR}"'"
SAM3_CKPT="'"${SAM3_CKPT}"'"
NUM_EPISODES="'"${NUM_EPISODES}"'"
NUM_SEEDS="'"${NUM_SEEDS}"'"
SAVE_VIDEOS="'"${SAVE_VIDEOS}"'"

for ITER in '"${CKPTS}"'; do
  CKPT_DIR="${CKPT_BASE_DIR}/${ITER}"
  PT_FILE="${CKPT_DIR}/model_ema_bf16.pt"

  echo ""
  echo "================================================================"
  echo "  Checkpoint: ${ITER}"
  echo "  Directory:  ${CKPT_DIR}"
  echo "  Mode:       CMR-only (4 datasets)"
  echo "================================================================"

  # Convert distcp → .pt if not already done
  if [ ! -f "${PT_FILE}" ]; then
    echo "  model_ema_bf16.pt not found — running convert_distcp_to_pt.py ..."
    python ./scripts/convert_distcp_to_pt.py "${CKPT_DIR}/model" "${CKPT_DIR}"
    if [ ! -f "${PT_FILE}" ]; then
      echo "  ERROR: conversion failed, ${PT_FILE} still missing. Skipping."
      continue
    fi
    echo "  Conversion complete: ${PT_FILE}"
  else
    echo "  model_ema_bf16.pt already exists, skipping conversion."
  fi

  # Run quantitative evaluation (CMR-only, no --test_episodes_json)
  LOGFILE="logs/scratch_quant_eval_cmr-only_${ITER}_$(date +%Y-%m-%d_%H-%M-%S).log"
  echo "  Running CMR-only evaluation → ${LOGFILE}"

  EVAL_CMD="CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
    --experiment ${EXPERIMENT} \
    --ckpt_path ${PT_FILE} \
    --ckpt_labels ${ITER} \
    --sam3_checkpoint ${SAM3_CKPT} \
    --num_episodes ${NUM_EPISODES} \
    --num_seeds ${NUM_SEEDS}"

  if [ "${SAVE_VIDEOS}" = "true" ]; then
    EVAL_CMD="${EVAL_CMD} --save_videos"
  fi

  eval "${EVAL_CMD}" 2>&1 | tee "${LOGFILE}"

  echo "  Done: ${ITER}"
done

echo ""
echo "================================================================"
echo "  All checkpoints processed (CMR-only)."
echo "================================================================"
'

