#!/usr/bin/env bash
# Run the JHU dVRK Mono Phase-2 self-forcing student in real-time on the
# local workstation, driven by ``scripts/keyboard/keyboard_controller.py``,
# served by the locally-built ``flashsim:local`` Docker image.
#
# What this script does
# ---------------------
#
#   1. Pre-flight: verify Docker, the local image, the SF checkpoint, the
#      X11 display, and the first-frame path.
#   2. Authorise the docker container to talk to the host X server (xhost
#      +local:docker), so the in-container ``cv2.imshow`` window and the
#      pynput keyboard listener can both reach the local display.
#   3. ``docker run`` ``flashsim:local`` with:
#        - ``--gpus 'device=1'`` (RTX 5880 Ada, idle; GPU 0 hosts Xorg)
#        - X11 passthrough (DISPLAY, XAUTHORITY, /tmp/.X11-unix)
#        - pynput requires libX11 access; ``--ipc=host`` and the X mounts
#          give us that without any container privilege flags.
#        - Bind-mounts:
#            * ``flashsim-jg``           -> /workspace/flashsim
#            * ``cosmos-h-surgical-simulator-rt`` -> /workspace/c-h-s-s (RO)
#            * ``$CHECKPOINTS_DIR``      -> /checkpoints (RO)
#            * ``$HF_HOME``              -> /root/.cache/huggingface
#        - Runtime ``pip install -e .[streaming,streaming_viewer] pynput``
#          on top of the image (idempotent; should be a no-op once the
#          editable install has been done once).
#   4. Launches ``projects.cosmos_h_surgical.run_keyboard`` against the JHU
#      Mono SF student config, at the 288x512 training resolution.
#   5. On exit (Esc, normal completion, or Ctrl-C) it drops the X auth
#      grant via ``xhost -local:docker`` so we don't leave the X server
#      open for long.
#
# Configuration via environment variables
# ---------------------------------------
#
#   FLASHSIM_REPO         (default /home/lzbinden/git/flashsim-jg)
#   CHSS_REPO             (default /home/lzbinden/git/cosmos-h-surgical-simulator-rt)
#   CHECKPOINTS_DIR       (default /home/lzbinden/checkpoints)
#   FLASHSIM_IMAGE        (default flashsim:local)
#   GPU_DEVICE            (default 1 - the RTX 5880 Ada; set to 0 for the
#                           A6000 if needed; "all" for both)
#   HF_TOKEN              (required for the first-time Cosmos-Reason1 +
#                           WanVAE downloads; export before running)
#   FIRST_FRAME           (required - JHU dVRK first frame path on host)
#
# Usage
# -----
#
#   export HF_TOKEN=<your hf token>
#   FIRST_FRAME=/path/to/jhu_first_frame.png \
#     scripts/keyboard/run_local_sim.sh
#
#   # Pass extra args straight through to run_keyboard.py:
#   FIRST_FRAME=/path/to/frame.png scripts/keyboard/run_local_sim.sh \
#     --keyboard_layout qwertz \
#     --save_video /tmp/run_$(date +%Y%m%d_%H%M%S).mp4
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
FLASHSIM_REPO="${FLASHSIM_REPO:-/home/lzbinden/git/flashsim-jg}"
CHSS_REPO="${CHSS_REPO:-/home/lzbinden/git/cosmos-h-surgical-simulator-rt}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/lzbinden/checkpoints}"
FLASHSIM_IMAGE="${FLASHSIM_IMAGE:-flashsim:local}"
GPU_DEVICE="${GPU_DEVICE:-1}"
HF_HOME_HOST="${HF_HOME:-${HOME}/.cache/huggingface}"
CHECKPOINT_RELPATH="sf-training_checkpoints/jhu_dvrk_mono_i4-sf_no_s3_resumable/checkpoints/iter_000001000/model_ema_bf16.pt"

# ---------------------------------------------------------------------------
# Pre-flight checks (fail fast before docker run)
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "[run_local_sim] docker is not installed or not on PATH." >&2
    exit 1
fi

if ! docker image inspect "$FLASHSIM_IMAGE" >/dev/null 2>&1; then
    echo "[run_local_sim] docker image '$FLASHSIM_IMAGE' not found." >&2
    echo "[run_local_sim]   Build it with:" >&2
    echo "[run_local_sim]     cd $FLASHSIM_REPO && docker build -t $FLASHSIM_IMAGE -f docker/Dockerfile ." >&2
    exit 1
fi

if [[ ! -d "$FLASHSIM_REPO" ]]; then
    echo "[run_local_sim] FLASHSIM_REPO=$FLASHSIM_REPO does not exist." >&2
    exit 1
fi
if [[ ! -d "$CHSS_REPO" ]]; then
    echo "[run_local_sim] CHSS_REPO=$CHSS_REPO does not exist." >&2
    exit 1
fi

CKPT_HOST="$CHECKPOINTS_DIR/$CHECKPOINT_RELPATH"
if [[ ! -f "$CKPT_HOST" ]]; then
    echo "[run_local_sim] Missing SF student checkpoint:" >&2
    echo "[run_local_sim]   $CKPT_HOST" >&2
    echo "[run_local_sim] Produce it on the cluster (Phase B of the plan):" >&2
    echo "[run_local_sim]   On the cluster:" >&2
    echo "[run_local_sim]     cd /lustre/.../cosmos-h-surgical-simulator-rt" >&2
    echo "[run_local_sim]     ./scripts/convert_distcp_to_pt.py \\" >&2
    echo "[run_local_sim]       /lustre/.../jhu_dvrk_mono_i4-sf_no_s3_resumable/checkpoints/iter_000001000 \\" >&2
    echo "[run_local_sim]       /lustre/.../jhu_dvrk_mono_i4-sf_no_s3_resumable/checkpoints/iter_000001000_export" >&2
    echo "[run_local_sim]   On this host (rsync mirroring the lustre tree under \$CHECKPOINTS_DIR):" >&2
    echo "[run_local_sim]     mkdir -p $CHECKPOINTS_DIR/sf-training_checkpoints/jhu_dvrk_mono_i4-sf_no_s3_resumable/checkpoints/iter_000001000" >&2
    echo "[run_local_sim]     scp <cluster>:.../iter_000001000_export/model_ema_bf16.pt $CKPT_HOST" >&2
    exit 1
fi

if [[ -z "${FIRST_FRAME:-}" ]]; then
    echo "[run_local_sim] FIRST_FRAME=<path> environment variable is required." >&2
    echo "[run_local_sim]   Pick a representative frame from a JHU dVRK test episode." >&2
    exit 1
fi
if [[ ! -f "$FIRST_FRAME" ]]; then
    echo "[run_local_sim] FIRST_FRAME=$FIRST_FRAME does not exist." >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[run_local_sim] WARNING: HF_TOKEN is not set." >&2
    echo "[run_local_sim]   The first run downloads Cosmos-Reason1 (~14 GB) +" >&2
    echo "[run_local_sim]   WanVAE checkpoints from Hugging Face and will fail" >&2
    echo "[run_local_sim]   without auth.  Subsequent runs reuse the cache at" >&2
    echo "[run_local_sim]   $HF_HOME_HOST and don't need the token." >&2
fi

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[run_local_sim] DISPLAY is not set - keyboard listener and cv2 window need an X11 display." >&2
    exit 1
fi

# Wayland-on-Xorg compatibility note: on Ubuntu 24.04 the GNOME default is
# Wayland; pynput needs X11.  If you logged in with the "Ubuntu on Xorg"
# session this is fine.  XDG_SESSION_TYPE = "x11" confirms.
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
    echo "[run_local_sim] WARNING: Wayland session detected." >&2
    echo "[run_local_sim]   pynput requires X11; either log out and pick" >&2
    echo "[run_local_sim]   'Ubuntu on Xorg' at the login screen, or run" >&2
    echo "[run_local_sim]   under a separate Xorg server / Xephyr." >&2
fi

# ---------------------------------------------------------------------------
# X11 passthrough setup
# ---------------------------------------------------------------------------
# ``xhost +local:docker`` adds the docker daemon as a local trusted client of
# the host X server so the container can render to it.  We tighten it back
# with ``-local:docker`` in the EXIT trap regardless of how the script ends.
if ! command -v xhost >/dev/null 2>&1; then
    echo "[run_local_sim] xhost not installed; install with: sudo apt install x11-xserver-utils" >&2
    exit 1
fi
# ``xhost +local:`` is the standard docker-on-X11 grant: it allows ANY local
# connection (including the container's root user) to talk to the host X
# server.  ``+local:docker`` would only grant a unix user named ``docker``,
# which most distros don't have, so the container's root ends up being
# rejected by xauth and cv2.imshow opens a window the user can never see.
# Using ``+local:`` is broader but limited to the local box (no remote X
# clients can connect via TCP regardless), which is fine for a single-user
# workstation.  We tighten it back with ``-local:`` in the EXIT trap.
xhost +local: >/dev/null
trap 'xhost -local: >/dev/null 2>&1 || true' EXIT

XAUTH_HOST="${XAUTHORITY:-${HOME}/.Xauthority}"
if [[ ! -f "$XAUTH_HOST" ]]; then
    echo "[run_local_sim] WARNING: XAUTHORITY=$XAUTH_HOST does not exist; container may fail to talk to X." >&2
fi

mkdir -p "$HF_HOME_HOST"

# Pass extra CLI flags through to run_keyboard.py.  Defaults below render at
# 288x512 (JHU training res), qwerty layout; override with env or args.
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# docker run
# ---------------------------------------------------------------------------
echo "[run_local_sim] launching flashsim container..."
echo "[run_local_sim]   image    : $FLASHSIM_IMAGE"
echo "[run_local_sim]   GPU      : $GPU_DEVICE"
echo "[run_local_sim]   ckpt     : $CKPT_HOST"
echo "[run_local_sim]   first_frame : $FIRST_FRAME"
echo "[run_local_sim]   DISPLAY  : $DISPLAY"

# pynput's evdev / Xlib backends need to read /proc/<pid>/exe of input
# events; we pass --ipc=host (already required by pytorch's shared-memory
# datasets) and the standard X11 mounts.  No --privileged.
docker run --rm \
    --gpus "device=$GPU_DEVICE" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    -e DISPLAY="$DISPLAY" \
    -e XAUTHORITY=/root/.Xauthority \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTHONPATH=/workspace/flashsim:/workspace/c-h-s-s \
    -e PYTHONUNBUFFERED=1 \
    -e QT_X11_NO_MITSHM=1 \
    -e QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$XAUTH_HOST":/root/.Xauthority:ro \
    -v "$FLASHSIM_REPO":/workspace/flashsim \
    -v "$CHSS_REPO":/workspace/c-h-s-s:ro \
    -v "$CHECKPOINTS_DIR":/checkpoints:ro \
    -v "$HF_HOME_HOST":/root/.cache/huggingface \
    -v "$FIRST_FRAME":/workspace/first_frame.png:ro \
    "$FLASHSIM_IMAGE" \
    bash -c '
        set -e
        cd /workspace/flashsim
        # Idempotent editable install + pynput.  This is fast on a warm
        # image (a few seconds) and free of network on subsequent runs.
        # Most of the deps are already in the image (see docker/Dockerfile).
        pip install -e ".[streaming,streaming_viewer]" pynput \
            --no-build-isolation --quiet
        # ``flashsim.distributed.init`` calls
        # ``dist.init_process_group(init_method="env://")`` which requires
        # RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT to be
        # set in the environment.  ``torchrun --standalone --nnodes=1
        # --nproc_per_node=1`` injects all of those for the single-GPU
        # case (this matches the launch convention of the existing
        # ``projects/cosmos_h_surgical/run.py``; see flashsim README).
        torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            -m projects.cosmos_h_surgical.run_keyboard \
            --first_frame_path /workspace/first_frame.png \
            --resolution 288,512 \
            "$@"
    ' bash "${EXTRA_ARGS[@]}"

echo "[run_local_sim] done."
