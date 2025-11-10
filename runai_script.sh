#!/bin/bash
set -euo pipefail

# ===== 配置区（按需改动）=====
JOB_NAME="isaaclab-docking"
PROJECT="htian"

# 使用你推到内网仓库的镜像
IMAGE="aicregistry:5000/htian/isaac-lab-custom:2.3.0"
IMAGE_PULL_SECRET=""             # 如内网仓库需要鉴权，填 secret 名；否则留空

GPUS=1
RUN_AS_USER=false                # NFS 权限需要时改为 true
GPU_NODE_TYPE="A100"             # 留空不限制

REPO_DIR="/nfs/home/htian/IsaacLab"
LOG_DIR="/nfs/home/htian/isaaclab_logs"
CACHE_BASE="/nfs/docker/isaac-sim/cache"
WORKDIR_IN="/workspace/IsaacLab"
# =================================

mkdir -p "$REPO_DIR" "$LOG_DIR" \
         "$CACHE_BASE/kit" "$CACHE_BASE/ov" "$CACHE_BASE/pip" \
         "$CACHE_BASE/glcache" "$CACHE_BASE/computecache"

# 先删同名任务（忽略失败）
runai delete job "$JOB_NAME" -p "$PROJECT" >/dev/null 2>&1 || true

# 组装 runai flags（所有 flag 必须在 --command 之前）
RUNAI_CMD=(
  runai submit
  --name "$JOB_NAME"
  -p "$PROJECT"
  -i "$IMAGE"
  -g "$GPUS"
  --large-shm
  --backoff-limit 0
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e PYTHONNOUSERSITE=1
  -v "$REPO_DIR":"$WORKDIR_IN":rw
  -v "$REPO_DIR":/workspace/isaaclab:rw
  -v "$CACHE_BASE/kit":/isaac-sim/kit/cache:rw
  -v "$CACHE_BASE/ov":/root/.cache/ov:rw
  -v "$CACHE_BASE/pip":/root/.cache/pip:rw
  -v "$CACHE_BASE/glcache":/root/.cache/nvidia/GLCache:rw
  -v "$CACHE_BASE/computecache":/root/.nv/ComputeCache:rw
  -v "$LOG_DIR":"$WORKDIR_IN"/logs:rw
  --working-dir "$WORKDIR_IN"
)

if [ "${RUN_AS_USER}" = true ]; then
  RUNAI_CMD+=(--run-as-user)
fi
if [ -n "${GPU_NODE_TYPE}" ]; then
  RUNAI_CMD+=(--node-type "${GPU_NODE_TYPE}")
fi
if [ -n "${IMAGE_PULL_SECRET}" ]; then
  RUNAI_CMD+=(--image-pull-secret "${IMAGE_PULL_SECRET}")
fi

# 容器内执行的命令：方案A（PYTHONPATH 置顶 + _isaac_sim/python.sh）
RUNAI_CMD+=( --command -- bash -lc '
  set -euo pipefail

  echo "[Init] Using mounted repo at: '"$WORKDIR_IN"'"
  ls -la '"$WORKDIR_IN"' | head -n 50

  # 使用 Omniverse 的 Python 包装器（推荐）
  PY="'"$WORKDIR_IN"'/_isaac_sim/python.sh"
  if [ ! -x "$PY" ]; then
    PY="/isaac-sim/kit/python/bin/python3"
  fi
  echo "[Info] Using python wrapper: $PY"

  # ✅ 覆盖导入顺序：把本地源码放到最前（在 set -u 下安全）
  export PYTHONPATH="'"$WORKDIR_IN"'/source:'"$WORKDIR_IN"'/source/isaaclab_tasks${PYTHONPATH:+:$PYTHONPATH}"

  # 自检：应看到本地源码路径在 sys.path 前列；命名空间包可能无 __file__，打印 spec 更稳
  $PY - <<PYCODE
import sys, importlib.util, pkgutil
print("PY:", sys.executable)
print("sys.path[:6]:", sys.path[:6])

spec = importlib.util.find_spec("isaaclab")
print("isaaclab spec:", spec)
if spec and getattr(spec, "origin", None):
    print("isaaclab origin:", spec.origin)
if spec and getattr(spec, "submodule_search_locations", None):
    print("isaaclab search_locations:", list(spec.submodule_search_locations))

try:
    tasks_spec = importlib.util.find_spec("isaaclab_tasks")
    print("isaaclab_tasks spec:", tasks_spec)
    if tasks_spec and getattr(tasks_spec, "origin", None):
        print("isaaclab_tasks origin:", tasks_spec.origin)
    if tasks_spec and getattr(tasks_spec, "submodule_search_locations", None):
        print("isaaclab_tasks search_locations:", list(tasks_spec.submodule_search_locations))
except Exception as e:
    print("isaaclab_tasks import check failed:", e)
PYCODE

  # 运行你的启动脚本或兜底命令（同样用 wrapper）
  if [ -f '"$WORKDIR_IN"'/runai_py.sh ]; then
    echo "[Run] bash runai_py.sh"
    bash '"$WORKDIR_IN"'/runai_py.sh
  else
    echo "[Warn] runai_py.sh 不存在，执行兜底训练命令（请按需改 task）"
    $PY scripts/reinforcement_learning/rl_games/train.py --task Isaac-Factory-PegInsert-Direct-v0
  fi
' )

# 提交任务
"${RUNAI_CMD[@]}"
