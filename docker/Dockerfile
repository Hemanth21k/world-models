# ═══════════════════════════════════════════════════════════════════════════════
# Stage: base
# Shared system dependencies, Python tooling, and GPU display setup.
# Build context: repo root  (docker build -f docker/Dockerfile .)
# ═══════════════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/pytorch:25.04-py3 AS base

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN apt-get update && apt-get install -y \
    # build tools
    build-essential yasm cmake libtool git pkg-config tmux \
    autoconf automake texinfo \
    # media / codec (ffmpeg + libav for video decoding in both models)
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libass-dev libfreetype6-dev libvorbis-dev \
    # OpenGL / display (GR00T-H simulation rendering)
    libegl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools && pip install uv

# EGL and Vulkan loader config (required for GR00T-H headless rendering)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.140"}}\n' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

WORKDIR /workspace


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: groot-h
# GR00T-H VLA model (3B, Cosmos-Reason-2B + diffusion transformer action head)
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS groot-h

# Fix known pip constraint conflicts present in the NVIDIA base image
RUN sed -i 's/dill==0.3.9/dill==0.3.8/g'     /etc/pip/constraint.txt 2>/dev/null || true && \
    sed -i 's/scipy==1.15.2/scipy==1.15.3/g' /etc/pip/constraint.txt 2>/dev/null || true

# Extra packages required by GR00T-H beyond its pyproject.toml
RUN pip install \
    imageio h5py boto3 \
    "transformers[torch]" deepspeed timm peft diffusers \
    wandb tianshou dm_tree openai \
    albumentations==1.4.18 decord==0.6.0 torchcodec==0.4.0

# pytorch3d — compiled from source (~10 min, cached after first build)
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && pip install .

# Install GR00T-H package (external_dependencies/ excluded via .dockerignore)
COPY vla/GR00T-H /workspace/groot_h
RUN cd /workspace/groot_h && uv sync --python 3.10 && uv pip install -e .


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: vjepa2
# V-JEPA 2 — self-supervised video world model (Meta FAIR)
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS vjepa2

# Install requirements before copying source for better layer caching
COPY latent_predictive/vjepa2/requirements.txt /tmp/vjepa2_requirements.txt
RUN pip install --no-cache-dir -r /tmp/vjepa2_requirements.txt

COPY latent_predictive/vjepa2 /workspace/vjepa2
RUN pip install --no-cache-dir -e /workspace/vjepa2

ENV PYTHONPATH=/workspace/vjepa2
ENV NCCL_DEBUG=WARN
ENV TORCH_NCCL_BLOCKING_WAIT=1
