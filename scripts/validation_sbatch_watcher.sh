#!/usr/bin/env bash
# Login-node-side watcher for the in-training validation quant-eval hook.
#
# The training callback (``EveryNValidationQuantEval`` in
# ``cosmos_predict2/_src/predict2/callbacks/validation_quant_eval.py``) drops
# fully-rendered ``iter_<iter>.ready.sbatch`` files into the queue dir.  This
# watcher polls that directory, submits each ready file via ``sbatch``, and
# renames the file based on the outcome:
#
#   iter_<iter>.ready.sbatch        -> picked up by this watcher
#   iter_<iter>.submitted.<jid>.sbatch -> sbatch returned 0; <jid> is the SLURM job id
#   iter_<iter>.failed.sbatch       -> sbatch returned non-zero; see <basename>.error
#
# The trainer never calls sbatch itself -- pyxis containers don't expose the
# SLURM client tools.  This script is the bridge.
#
# Usage (run on the SLURM submit/login node, ideally in a tmux window):
#
#   ./scripts/validation_sbatch_watcher.sh \
#       /lustre/.../cosmos-h-surgical-simulator-rt/validation/_sbatch/
#
# Optional:
#   --interval=N      Polling interval in seconds (default 30)
#   --once            Submit any pending files and exit (no loop)
#   --dry-run         Show what would be submitted; do not actually call sbatch
#   --quiet           Suppress per-poll status lines (errors / submissions still printed)
#
# The watcher is safe to kill and restart at any time -- pending files are
# picked up on the next poll.  Already-submitted / failed files are never
# re-submitted because their suffix changed.

set -euo pipefail

# --- arg parsing ---
INTERVAL=30
ONCE=0
DRY_RUN=0
QUIET=0
QUEUE_DIR=""

while (( $# )); do
  case "$1" in
    --interval=*) INTERVAL="${1#*=}";;
    --interval)   shift; INTERVAL="$1";;
    --once)       ONCE=1;;
    --dry-run)    DRY_RUN=1;;
    --quiet)      QUIET=1;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    -*) echo "unknown flag: $1" >&2; exit 2;;
    *)  QUEUE_DIR="$1";;
  esac
  shift
done

if [[ -z "$QUEUE_DIR" ]]; then
  echo "Usage: $0 <queue_dir> [--interval=N] [--once] [--dry-run] [--quiet]" >&2
  exit 2
fi
if [[ ! -d "$QUEUE_DIR" ]]; then
  echo "queue_dir does not exist: $QUEUE_DIR" >&2
  exit 2
fi
if (( ! DRY_RUN )) && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not on PATH; this script must run on the SLURM submit node." >&2
  exit 2
fi

QUEUE_DIR="$(cd "$QUEUE_DIR" && pwd)"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*"; }
log_quiet() { (( QUIET )) || log "$*"; }

trap 'log "watcher: SIGINT received, exiting."; exit 0' INT TERM

log "watcher: polling $QUEUE_DIR every ${INTERVAL}s  (dry_run=$DRY_RUN  once=$ONCE)"

submit_one() {
  local file="$1"
  local base
  base="$(basename "$file" .ready.sbatch)"   # iter_000001000

  if (( DRY_RUN )); then
    log "DRY-RUN: would submit $file"
    mv "$file" "$QUEUE_DIR/${base}.dryrun.sbatch"
    return 0
  fi

  local out err rc=0
  out="$(sbatch "$file" 2>&1)" || rc=$?
  if (( rc == 0 )); then
    # sbatch typically prints: "Submitted batch job 12345"
    local jid
    jid="$(printf '%s\n' "$out" | awk '/Submitted batch job/ {print $4; exit}')"
    if [[ -z "$jid" ]]; then jid="unknown"; fi
    log "submitted: $base -> job $jid"
    mv "$file" "$QUEUE_DIR/${base}.submitted.${jid}.sbatch"
  else
    log "FAILED: $base (rc=$rc): $out"
    {
      echo "[$(ts)] sbatch rc=$rc"
      echo "$out"
    } > "$QUEUE_DIR/${base}.error"
    mv "$file" "$QUEUE_DIR/${base}.failed.sbatch"
  fi
}

poll_once() {
  shopt -s nullglob
  local files=( "$QUEUE_DIR"/*.ready.sbatch )
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    log_quiet "no pending files."
    return 0
  fi
  # Sort by iter (lexicographic on iter_NNNNNNNNN works because of zero padding).
  IFS=$'\n' files=( $(printf '%s\n' "${files[@]}" | sort) )
  unset IFS
  log "found ${#files[@]} pending file(s); submitting in order."
  for f in "${files[@]}"; do
    submit_one "$f"
  done
}

if (( ONCE )); then
  poll_once
  exit 0
fi

while true; do
  poll_once
  sleep "$INTERVAL"
done
