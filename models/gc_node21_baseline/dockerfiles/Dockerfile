FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install node21 baseline algorithm
#  - Git clone to tmp directory (main branch, tag v1.1)
#  - Extract relevant files to /opt/algorithm/
#  - Remove tmp directory
RUN git clone --depth 1 --branch v1.1 https://github.com/node21challenge/node21_detection_baseline.git /tmp/algorithm && \
    mkdir -p /opt/algorithm && \
    mv /tmp/algorithm/process.py /opt/algorithm/process.py && \
    mv /tmp/algorithm/postprocessing.py /opt/algorithm/postprocessing.py && \
    mv /tmp/algorithm/training_utils /opt/algorithm/training_utils && \
    mv /tmp/algorithm/model.pth /opt/algorithm/model.pth && \
    mkdir -p /opt/algorithm/hub/checkpoints && \
    mv /tmp/algorithm/resnet50-19c8e357.pth /opt/algorithm/hub/checkpoints/resnet50-19c8e357.pth && \
    rm -rf /tmp/algorithm

# Install Node21 additional dependencies
RUN pip3 install --no-cache-dir evalutils==0.2.4

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_node21_baseline ${MHUB_MODELS_REPO}

# Configure app /opt/algorithm to be the torch hub directory to pick up the model weights correctly
ENV TORCH_HOME /opt/algorithm

# Add Node21 source path to the PYTHONPATH to be able to import the processor code
ENV PYTHONPATH "/app:/opt/algorithm"

# Specify default entrypoint
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_node21_baseline/config/default.yml"]
