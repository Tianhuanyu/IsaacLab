#!/bin/bash
set -euo pipefail

# ===== 配置区 =====
JOB_NAME="isaaclab-docking"
PROJECT="htian"
IMAGE="nvcr.io/nvidia/isaac-lab:2.3.0"   # 预构建 Isaac Lab（headless, 含sm_70内核）
GPUS=1
RUN_AS_USER=false          # Isaac Sim binaries require root to execute; flip to true only if NFS permissions demand it

REPO_DIR="/nfs/home/htian/IsaacLab"      # 你的 IsaacLab 代码（宿主机）
LOG_DIR="/nfs/home/htian/isaaclab_logs"  # 日志目录（宿主机）
CACHE_BASE="/nfs/docker/isaac-sim/cache" # 缓存根目录（宿主机）
GPU_NODE_TYPE="A100"                     # 留空默认调度任意GPU节点；示例值强制 A100
NODE_NAME=""                             # Run:AI CLI 暂不支持直接按主机名指定，留空避免报错
# ==================

# 确保宿主目录存在
mkdir -p "$REPO_DIR" "$LOG_DIR" \
         "$CACHE_BASE/kit" "$CACHE_BASE/ov" "$CACHE_BASE/pip" \
         "$CACHE_BASE/glcache" "$CACHE_BASE/computecache"

# 先删同名任务（忽略失败）
runai delete job "$JOB_NAME" -p "$PROJECT" >/dev/null 2>&1 || true

# 提交
RUNAI_CMD=(
  runai submit
  --name "$JOB_NAME"
  -p "$PROJECT"
  -i "$IMAGE"
  -g "$GPUS"
)
if [ "${RUN_AS_USER}" = true ]; then
  RUNAI_CMD+=(--run-as-user)
fi
if [ -n "${GPU_NODE_TYPE}" ]; then
  RUNAI_CMD+=(--node-type "${GPU_NODE_TYPE}")
fi
RUNAI_CMD+=(
  --large-shm
  --backoff-limit 0
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -v "$REPO_DIR":/workspace/IsaacLab:rw
  -v "$CACHE_BASE/kit":/isaac-sim/kit/cache:rw
  -v "$CACHE_BASE/ov":/root/.cache/ov:rw
  -v "$CACHE_BASE/pip":/root/.cache/pip:rw
  -v "$CACHE_BASE/glcache":/root/.cache/nvidia/GLCache:rw
  -v "$CACHE_BASE/computecache":/root/.nv/ComputeCache:rw
  -v "$LOG_DIR":/workspace/IsaacLab/logs:rw
  --working-dir /workspace/IsaacLab
  --command
  --
  bash
  -lc
  'cd /workspace/IsaacLab && bash runai_py.sh'
)

"${RUNAI_CMD[@]}"
