FROM mhubai/base:latest

# Update authors label
LABEL authors="s.vandeleemput@radboudumc.nl"

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# build/install CUDA 11 toolkit cudnn 8 and tensorflow 2.11.0 with GPU support (without conda)
RUN pip3 install --no-cache-dir \
    nvidia-cuda-runtime-cu11 \
    nvidia-cusolver-cu11 \
    nvidia-curand-cu11 \
    nvidia-cufft-cu11 \
    nvidia-cublas-cu11 \
    nvidia-cusparse-cu11 \
    nvidia-cudnn-cu11 \
    tensorflow==2.11.0 \
    nvidia-tensorrt==7.2.3.4 \
    --extra-index-url https://pypi.ngc.nvidia.com

# Configure required paths for tensorflow with GPU support
ENV NVIDIA_DIR /usr/local/lib/python3.8/dist-packages/nvidia
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/tensorrt:$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cuda_runtime/lib:$NVIDIA_DIR/cudnn/lib:$NVIDIA_DIR/cufft/lib:$NVIDIA_DIR/curand/lib:$NVIDIA_DIR/cusolver/lib:$NVIDIA_DIR/cusparse/lib

# Clone the main branch of MHubAI/models TODO enable
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_luna22_ismi_baseline" \
#    && git fetch https://github.com/MHubAI/models.git main

# Install luna22 ismi baseline algorithm and model weights
RUN git clone --depth 1 https://github.com/DIAGNijmegen/bodyct-luna22-ismi-algorithm-baseline.git /opt/algorithm

# Install algorithm requirements
RUN pip3 install --no-cache-dir evalutils==0.3.1

# Set PYTHONPATH to include algorithm code
ENV PYTHONPATH "/app:/opt/algorithm"

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_luna22_ismi_baseline/config/default.yml"]
