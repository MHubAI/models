FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install ASAP 2.1
RUN apt-get update && \
    apt-get -y install libpython3.8-dev && \
    curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb" && \
    dpkg --install ASAP-2.1-py38-Ubuntu2004.deb || true && \
    apt-get -f install --fix-missing --fix-broken --assume-yes && \
    ldconfig -v && \
    apt-get clean && \
    rm ASAP-2.1-py38-Ubuntu2004.deb

# Setup and install algorithm pipenv environment
#   1. Ensure we configure a new empty pipenv for Python 3.8
#   2. Link ASAP libraries into our environment
#   3. Upgrade pip
#   4. Upgrade version of numpy and numba to function correctly with ASAP
#   5. Install required dependencies for algorithm
RUN uv venv --python 3.8 .venv38 && \
    echo "/opt/ASAP/bin" > /app/.venv38/lib/python3.8/site-packages/asap.pth && \
    uv pip install -n -p .venv38 --upgrade numpy==1.24.4 numba==0.58.1 && \
    uv pip install -n -p .venv38 scipy==1.10.1 scikit-image==0.21.0 h5py==3.11.0

# build/install Tensorflow 2.11.0 with GPU support (without conda), with CUDA 11 toolkit and cudnn 8
# tensorflow-2.11.0 Python 3.7-3.10 cuDNN >= 8.1 CUDA >= 11.2
RUN uv pip install -n -p .venv38 pip \
 && uv run -p .venv38 pip install --no-cache-dir \
    nvidia-cuda-runtime-cu11 \
    nvidia-cusolver-cu11 \
    nvidia-curand-cu11 \
    nvidia-cufft-cu11 \
    nvidia-cublas-cu11 \
    nvidia-cusparse-cu11 \
    nvidia-cudnn-cu11 \
    nvidia-tensorrt==7.2.3.4 \
    tensorflow==2.11.0 \
    --extra-index-url https://pypi.ngc.nvidia.com

# Configure required paths for tensorflow with GPU support
ENV NVIDIA_DIR /app/.venv38/lib/python3.8/site-packages/nvidia
ENV LD_LIBRARY_PATH /app/.venv38/lib/python3.8/site-packages/tensorrt:$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cuda_runtime/lib:$NVIDIA_DIR/cudnn/lib:$NVIDIA_DIR/cufft/lib:$NVIDIA_DIR/curand/lib:$NVIDIA_DIR/cusolver/lib:$NVIDIA_DIR/cusparse/lib

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_wsi_bgseg ${MHUB_MODELS_REPO}

# Install pathology tissue background segmentation processor code and weights (version 1.0.0)
RUN git clone --depth 1 --branch 1.0.0 https://github.com/DIAGNijmegen/pathology-tissue-background-segmentation-processor.git /app/src

# Add model and algorithm code bases to python path
ENV PYTHONPATH="/app:/app/src"

# DEV
COPY models/gc_wsi_bgseg/utils /app/models/gc_wsi_bgseg/utils

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_wsi_bgseg/config/default.yml"]
