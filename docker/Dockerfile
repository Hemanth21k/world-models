# ═══════════════════════════════════════════════════════════════════════════════
# Stage: base
# Shared system dependencies, Python tooling, GPU display config.
# Build context must be the repo root:
#   docker build -f docker/Dockerfile --target <stage> .
# ═══════════════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/pytorch:25.04-py3 AS base

ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN apt-get update && apt-get install -y \
    # build tools
    build-essential yasm cmake libtool git pkg-config tmux \
    autoconf automake texinfo \
    # media / codec (ffmpeg + libav for torchcodec / decord / vjepa2)
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
    libass-dev libfreetype6-dev libvorbis-dev \
    # OpenGL / EGL (GR00T-H headless rendering + vjepa2 CV deps)
    libegl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools && pip install uv

# EGL + Vulkan loader config for headless GPU rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.140"}}\n' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

WORKDIR /workspace


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: groot-h-deps  [DEVELOPMENT target]
# All GR00T-H Python dependencies pre-installed. No source code included.
# Mount the repo at runtime for live editing:
#   docker run ... -v ./vla/GR00T-H:/workspace/groot_h world-models:groot-h-dev
# Then inside the container: pip install -e /workspace/groot_h --no-deps
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS groot-h-deps

# Fix known pip constraint conflicts in the NVIDIA base image
RUN sed -i 's/dill==0.3.9/dill==0.3.8/g'     /etc/pip/constraint.txt 2>/dev/null || true && \
    sed -i 's/scipy==1.15.2/scipy==1.15.3/g' /etc/pip/constraint.txt 2>/dev/null || true

# Additional packages GR00T-H needs beyond its pyproject.toml
RUN pip install \
    imageio h5py boto3 \
    "transformers[torch]" deepspeed timm peft diffusers \
    wandb tianshou dm_tree openai \
    albumentations==1.4.18 decord==0.6.0 torchcodec==0.4.0

# pytorch3d — compiled from source (~10 min, layer-cached after first build)
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && pip install .

# Pre-install GR00T-H's declared dependencies from the lockfile,
# without installing the gr00t package itself (no source needed yet).
COPY vla/GR00T-H/pyproject.toml vla/GR00T-H/uv.lock /tmp/groot_h_pkg/
RUN cd /tmp/groot_h_pkg && \
    uv export --frozen --no-dev --no-emit-project -o /tmp/groot_h_requirements.txt && \
    pip install -r /tmp/groot_h_requirements.txt

WORKDIR /workspace/groot_h


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: groot-h  [DEPLOYMENT target]
# Self-contained — source code baked in, ready to run without any mounts.
# ═══════════════════════════════════════════════════════════════════════════════
FROM groot-h-deps AS groot-h

# external_dependencies/ is excluded by .dockerignore (saves ~765 MB)
COPY vla/GR00T-H /workspace/groot_h
RUN pip install -e /workspace/groot_h --no-deps


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: vjepa2-deps  [DEVELOPMENT target]
# All V-JEPA 2 Python dependencies pre-installed. No source code included.
# Mount the repo at runtime for live editing:
#   docker run ... -v ./latent_predictive/vjepa2:/workspace/vjepa2 world-models:vjepa2-dev
# Then inside the container: pip install -e /workspace/vjepa2 --no-deps
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS vjepa2-deps

COPY latent_predictive/vjepa2/requirements.txt /tmp/vjepa2_requirements.txt
RUN pip install --no-cache-dir -r /tmp/vjepa2_requirements.txt

ENV PYTHONPATH=/workspace/vjepa2
ENV NCCL_DEBUG=WARN
ENV TORCH_NCCL_BLOCKING_WAIT=1

WORKDIR /workspace/vjepa2


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: vjepa2  [DEPLOYMENT target]
# Self-contained — source code baked in.
# ═══════════════════════════════════════════════════════════════════════════════
FROM vjepa2-deps AS vjepa2

COPY latent_predictive/vjepa2 /workspace/vjepa2
RUN pip install --no-cache-dir -e /workspace/vjepa2 --no-deps
