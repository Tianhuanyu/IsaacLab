# Dockerfile
ARG BASE_IMAGE=aicregistry:5000/ml/isaac-lab:2.3.0
FROM ${BASE_IMAGE}

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER=app

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    DEBIAN_FRONTEND=noninteractive

RUN groupadd -g ${GROUP_ID} ${USER} \
 && useradd  -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER}

RUN apt-get update && apt-get install -y --no-install-recommends \
      git vim less ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
