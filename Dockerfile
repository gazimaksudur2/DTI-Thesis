# DTIWork reproducible image: CUDA-capable Linux runtime with the
# `dti_research` conda environment baked in. Works on:
#   - hosts with an NVIDIA GPU (e.g. RTX 4060 Ti) via docker-compose's
#     `deploy.resources.reservations.devices` block,
#   - CPU-only hosts via `--device cpu` at the application level (XGBoost
#     and the rest of the stack are CPU-safe; only the CUDA runtime
#     libraries go unused).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    CONDA_ENV=dti_research

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

RUN conda create -y -n ${CONDA_ENV} python=3.11 \
    && conda install -y -n ${CONDA_ENV} -c conda-forge pytdc \
    && conda clean -afy

WORKDIR /workspace

COPY requirements.txt requirements-gpu.txt /tmp/

RUN /opt/conda/envs/${CONDA_ENV}/bin/pip install --no-cache-dir \
        -r /tmp/requirements.txt \
        -r /tmp/requirements-gpu.txt

ENV PATH=/opt/conda/envs/${CONDA_ENV}/bin:$PATH

CMD ["bash"]
