#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
#
## install dependencies
#RUN apt-get update
#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
#RUN apt-get install -y wget git build-essential
##RUN apt-add-repository universe
#RUN apt-get -y install software-properties-common
#RUN apt-get -y install pdsh
#RUN apt-get -y install python3-pip
#RUN apt-get -y install ninja-build
#RUN pip install pybind11
#RUN pip install python-config
#RUN pip install deepspeed
#RUN pip install regex
#
## run deepspeed
#COPY . .
#CMD ["bash", "scripts/fixed_global_bsz_run_scripts/fixed_global_bsz.sh"]

FROM nvidia/cuda:11.0-devel-ubuntu18.04

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update && \
    apt-get install -y git && \
    git --version

##############################################################################
# Client Liveness & Uncomment Port 22 for SSH Daemon
##############################################################################
# Keep SSH client alive from server side
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

##############################################################################
# Mellanox OFED
##############################################################################
ENV MLNX_OFED_VERSION=4.6-1.0.1.1
RUN apt-get install -y libnuma-dev
RUN cd ${STAGE_DIR} && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
    ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*

##############################################################################
# nv_peer_mem
##############################################################################
ENV NV_PEER_MEM_VERSION=1.1
ENV NV_PEER_MEM_TAG=1.1-0
RUN mkdir -p ${STAGE_DIR} && \
    git clone https://github.com/Mellanox/nv_peer_memory.git --branch ${NV_PEER_MEM_TAG} ${STAGE_DIR}/nv_peer_memory && \
    cd ${STAGE_DIR}/nv_peer_memory && \
    ./build_module.sh && \
    cd ${STAGE_DIR} && \
    tar xzf ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_VERSION}.orig.tar.gz && \
    cd ${STAGE_DIR}/nvidia-peer-memory-${NV_PEER_MEM_VERSION} && \
    apt-get update && \
    apt-get install -y dkms && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ${STAGE_DIR}/nvidia-peer-memory_${NV_PEER_MEM_TAG}_all.deb

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.1
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

# Install OpenSSH for MPI to communicate between containers
#RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
#    mkdir -p /var/run/sshd
# Allow OpenSSH to talk to containers without asking for confirmation
#RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
#    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
#    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py && \
    pip install --upgrade pip && \
    # Print python an pip version
    python -V && pip -V
RUN pip install pyyaml
RUN pip install ipython

###############################################################################
## TensorFlow
###############################################################################
#ENV TENSORFLOW_VERSION=1.15.2
#RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION}

##############################################################################
# Some Packages
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen
RUN pip install psutil \
                yappi \
                cffi \
                ipdb \
                pandas \
                matplotlib \
                py3nvml \
                pyarrow \
                graphviz \
                astor \
                boto3 \
                tqdm \
                sentencepiece \
                msgpack \
                requests \
                pandas \
                sphinx \
                sphinx_rtd_theme \
                scipy \
                numpy \
                sklearn \
                scikit-learn \
                nvidia-ml-py3 \
                mpi4py \
                cupy-cuda110

##############################################################################
## SSH daemon port inside container cannot conflict with host OS port
###############################################################################
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > ${STAGE_DIR}/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

##############################################################################
# PyTorch
##############################################################################
#ENV PYTORCH_VERSION=1.7.0
#ENV TORCHVISION_VERSION=0.7.0
ENV TENSORBOARDX_VERSION=2.1
#RUN pip install torch==${PYTORCH_VERSION}
#RUN pip install torchvision==${TORCHVISION_VERSION}
RUN pip install tensorboardX==${TENSORBOARDX_VERSION}
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
    rm -rf /usr/lib/python3/dist-packages/PyYAML-*

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
#RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
#RUN usermod -aG sudo deepspeed
#RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
#USER deepspeed
#USER root

##############################################################################
# Apex
##############################################################################
RUN pip install regex

RUN sudo git clone https://github.com/NVIDIA/apex
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

##############################################################################
# NCCL
##############################################################################
#RUN git clone https://github.com/NVIDIA/nccl.git ${STAGE_DIR}/nccl
#RUN cd ${STAGE_DIR}/nccl && make -j src.build
#RUN cd ${STAGE_DIR}/nccl && \
#    sudo apt-get install -y build-essential devscripts debhelper fakeroot && \
#    make pkg.debian.build && \
#    ls build/pkg/deb/

##############################################################################
# DeepSpeed
##############################################################################
#RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed
COPY DeepSpeed ${STAGE_DIR}/DeepSpeed
RUN cd ${STAGE_DIR}/DeepSpeed && pip install .
RUN python -c "import deepspeed; print(deepspeed.__version__)"


# run deepspeed
#COPY . /megatron
#WORKDIR /megatron
COPY DSE .
#COPY /h/aqiao/megatron/preprocessed_data .
CMD ["bash", "scripts/fixed_global_bsz_run_scripts/fixed_global_bsz.sh"]

