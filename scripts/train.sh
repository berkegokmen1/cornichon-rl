#!/usr/bin/env bash
# Start (or resume) a PPO run in a detached tmux session named after the run.
#   scripts/train.sh GPU NAME [extra args]          e.g. scripts/train.sh 0 ppo_v1
#   scripts/train.sh 0 ppo_v1 --resume              continue out_dir/ppo_v1/latest.pt
#   CONFIG=configs/other.yaml scripts/train.sh 0 x ppo.lr=1e-4
# GPU is the physical index; inside the process it is cuda:0. Check who is on it first (printed below).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU=$1
NAME=$2
shift 2
CONFIG=${CONFIG:-configs/default.yaml}
# The venv lives on node-local disk, so its path depends on the node.
for candidate in "${CORNICHON_PY:-}" /opt/berke/envs/cornichon-rl/bin/python /data/local/berke/envs/cornichon-rl/bin/python; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then PY=$candidate; break; fi
done
: "${PY:?no cornichon-rl venv on this node (see README_RL.md)}"
[ -x "$ROOT/headless/build/install/headless/bin/headless" ] || { echo "build the simulator: ./gradlew :headless:installDist"; exit 1; }

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -i "$GPU"
LOG_DIR=/data/local/berke/cornichon-rl/logs
mkdir -p "$LOG_DIR"
tmux new-session -d -s "$NAME" \
  "cd $ROOT && CUDA_VISIBLE_DEVICES=$GPU $PY -m cornichon_rl.ppo --config $CONFIG --name $NAME $* 2>&1 | tee -a $LOG_DIR/$NAME.log"
echo "tmux session '$NAME' on $(hostname) GPU $GPU; log $LOG_DIR/$NAME.log"
