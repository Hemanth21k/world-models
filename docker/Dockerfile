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
    # media / codec (ffmpeg + full libav dev headers for PyAV / torchcodec / decord)
    ffmpeg libavcodec-dev libavformat-dev libavdevice-dev \
    libavfilter-dev libavutil-dev libswscale-dev libswresample-dev \
    libass-dev libfreetype6-dev libvorbis-dev \
    # OpenGL / EGL (GR00T-H headless rendering + vjepa2 CV deps)
    libegl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
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
# Python 3.10 venv with all GR00T-H dependencies. No source code included.
#
# GR00T-H requires Python 3.10 (pyproject.toml: requires-python = "==3.10.*").
# uv manages its own Python installation, so it downloads 3.10 and creates a
# venv at /workspace/groot_h/.venv regardless of the system Python (3.12).
#
# For development (live mount):
#   docker run ... -v ./vla/GR00T-H:/workspace/groot_h world-models:groot-h-dev
#   Inside: uv sync --python 3.10   (installs gr00t into the existing venv)
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS groot-h-deps

# Copy only the lockfiles to create a cacheable deps layer (no source yet).
# --no-install-workspace: skip installing gr00t and workspace members —
#   source isn't available at this stage.
COPY vla/GR00T-H/pyproject.toml vla/GR00T-H/uv.lock /workspace/groot_h/
WORKDIR /workspace/groot_h
RUN uv sync --python 3.10 --no-install-workspace

# Put the Python 3.10 venv on PATH so all subsequent steps use it
ENV PATH="/workspace/groot_h/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/groot_h/.venv"

# Extra packages not in pyproject.toml but used at runtime
RUN pip install \
    imageio h5py boto3 deepspeed tianshou openai \
    decord==0.6.0

# GR00T-H's Eagle backbone requires FlashAttention2 at inference time.
RUN uv pip install --no-build-isolation flash-attn==2.7.4.post1

# pytorch3d compiled for Python 3.10 (pip now points to the venv)
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && pip install .


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: groot-h  [DEPLOYMENT target]
# Self-contained — source baked in, venv completed, ready to run.
# ═══════════════════════════════════════════════════════════════════════════════
FROM groot-h-deps AS groot-h

# external_dependencies/ excluded by .dockerignore (saves ~765 MB)
COPY vla/GR00T-H /workspace/groot_h
# uv detects the existing venv and installs the gr00t project into it
RUN uv sync --python 3.10 && \
    uv pip install --no-build-isolation flash-attn==2.7.4.post1


# ═══════════════════════════════════════════════════════════════════════════════
# Stage: vjepa2-deps  [DEVELOPMENT target]
# All V-JEPA 2 Python dependencies pre-installed. No source code included.
#
# For development (live mount):
#   docker run ... -v ./latent_predictive/vjepa2:/workspace/vjepa2 world-models:vjepa2-dev
#   Inside: pip install -e . --no-deps
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
