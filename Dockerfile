FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
        python3 \
        python3-venv \
        python3-pip \
        git \
        curl \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
RUN echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list
RUN apt-get update -y
RUN apt-get install -y nvhpc-24-5

ENV NVARCH=Linux_x86_64
ENV NVCOMPILERS=/opt/nvidia/hpc_sdk
ENV MANPATH=$NVCOMPILERS/$NVARCH/24.5/compilers/man
ENV PATH=$NVCOMPILERS/$NVARCH/24.5/compilers/bin:$PATH
ENV PATH=$NVCOMPILERS/$NVARCH/24.5/comm_libs/mpi/bin:$PATH
ENV MANPATH=$NVCOMPILERS/$NVARCH/24.5/comm_libs/mpi/man

ENV DEVITO_PLATFORM=nvidiaX
ENV DEVITO_COMPILER=pgcc
ENV DEVITO_LANGUAGE=openacc

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY ./UltraWave /app/UltraWave

RUN ls -la
WORKDIR /app/UltraWave

RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

WORKDIR /app