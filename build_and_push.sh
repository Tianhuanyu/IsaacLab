#!/usr/bin/env bash
set -euo pipefail

# ===== 配置 =====
REG="aicregistry:5000"      # 你们内网 registry
NS="htian"                  # 命名空间
OUT_NAME="isaac-lab-custom" # 输出镜像名
TAG="2.3.0"                 # 版本
OUT_IMG="${REG}/${NS}/${OUT_NAME}:${TAG}"

# 作为基础镜像使用（必须是“本机已存在”或“内网可拉取”的地址）
BASE_IMAGE="${REG}/ml/isaac-lab:${TAG}"
# 可改成你们实际可用的地址，例如:
# BASE_IMAGE="${REG}/htian/isaac-lab-base:${TAG}"
# 或者如果已经在本机有缓存：BASE_IMAGE="nvcr.io/nvidia/isaac-lab:${TAG}"

# ===== 检查基础镜像是否存在于本机 =====
if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "[ERR] 本机没有基础镜像: ${BASE_IMAGE}"
  echo "      请选择其一："
  echo "      1) 让管理员提供内网基础镜像，并把 BASE_IMAGE 改成该地址。"
  echo "      2) 若你手上有 tar 包：docker load -i isaac-lab-${TAG}.tar，然后把 BASE_IMAGE 指向加载出的镜像名。"
  echo "      3) 在可访问外网的机器拉取/保存，再拷到这里："
  echo "         # 外网机:"
  echo "         docker pull nvcr.io/nvidia/isaac-lab:${TAG}"
  echo "         docker save nvcr.io/nvidia/isaac-lab:${TAG} -o isaac-lab-${TAG}.tar"
  echo "         # 传到本机后:"
  echo "         docker load -i isaac-lab-${TAG}.tar"
  echo "         docker tag nvcr.io/nvidia/isaac-lab:${TAG} ${BASE_IMAGE}"
  exit 1
fi

echo "[OK] 使用基础镜像: ${BASE_IMAGE}"

# ===== 构建 =====
docker build . -f Dockerfile \
  --tag "${OUT_IMG}" --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  --build-arg USER="${USER}"

# ===== 推送到内网仓库 =====
# 如需登录内网仓库，请先：docker login ${REG}
docker push "${OUT_IMG}"

echo "[DONE] Pushed ${OUT_IMG}"
docker images | grep "${OUT_NAME}" || true
