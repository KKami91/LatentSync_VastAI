FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 전체 프로젝트 복사 (ComfyUI 포함)
COPY . /workspace

# requirements.txt 기반 의존성 설치
RUN pip3 install --no-cache-dir -r requirements.txt

# Huggingface CLI 설치 및 모델 다운로드
RUN pip3 install --upgrade huggingface-hub
RUN --mount=type=cache,target=/root/.cache/huggingface \
    huggingface-cli download ByteDance/LatentSync \
    --local-dir custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints \
    --exclude "*.git*" "README.md" || echo "Hugging Face 모델 다운로드 실패: 캐시 문제, 네트워크 문제 또는 모델 이름 확인 필요"

# runpod serverless 실행을 위한 진입점 설정
CMD ["python3", "LatentSync_basic.py"]