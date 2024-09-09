FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && \
    apt install -y --no-install-recommends git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Install spider baseline algorithm
#  - Git clone to tmp directory and reset to specific commit (main branch commit at 2023-09-12)
#  - Extract relevant files to src
#  - Remove tmp directory to remove history from layer
RUN git clone https://github.com/DIAGNijmegen/SPIDER-Baseline-IIS.git /tmp/algorithm && \
    cd /tmp/algorithm && \
    git reset --hard 1b26e2e0a710151db66f1cac0b51a3408368a2ef && \
    mkdir -p /app/src/datasets/spider_input/images && \
    mv /tmp/algorithm/devel /app/src/devel && \
    mv /tmp/algorithm/experiments /app/src/experiments && \
    rm -rf /tmp/algorithm

# Install additional spider baseline dependencies (stable branch commit at 2023-09-12)
RUN pip3 install --no-cache-dir git+https://github.com/DIAGNijmegen/Tiger.git@4c63f6d58ed3f8e7a0cec04b0352949f7b512919

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_spider_baseline ${MHUB_MODELS_REPO}

# Setup required environment variables for SPIDER algorithm
ENV PYTHONPATH="/app:/app/src/devel"
ENV VERSEG_BASEDIR="/app/src/"

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_spider_baseline/config/default.yml"]
