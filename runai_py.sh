#!/bin/bash
set -euo pipefail
set -x   # 打印每条命令，便于看 logs

cd /workspace/IsaacLab

# 0) 补回软链：让 ./isaaclab.sh 能找到 /isaac-sim/python.sh
if [ ! -e _isaac_sim ]; then
  ln -s /isaac-sim _isaac_sim
fi

# 1) 基本自检（不会因失败而退出）
echo "[INFO] PWD=$(pwd)  whoami=$(whoami)"
ls -la || true
/isaac-sim/python.sh -V || true
nvidia-smi || true

# 2) 如未安装 rl_games，则尝试安装（已安装则跳过）
/isaac-sim/python.sh - <<'PY'
import importlib, sys
sys.exit(0 if importlib.util.find_spec("rl_games") else 1)
PY
if [ $? -ne 0 ]; then
  echo "[INFO] rl_games not found; installing via ./isaaclab.sh -i rl_games"
  ./isaaclab.sh -i rl_games || { echo "[ERROR] install rl_games failed (network blocked?)"; exit 20; }
fi

# 3) 开训（headless）
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Factory-PegInsert-Direct-v0 \
  --headless \
  --num_envs 2048 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
