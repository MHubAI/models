FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install spider baseline algorithm
#  - Git shallow clone to tmp directory
#  - Extract relevant files to src
#  - Remove tmp directory
RUN git clone --depth 1 https://github.com/DIAGNijmegen/SPIDER-Baseline-IIS.git /tmp/algorithm && \
    mkdir -p src/datasets/spider_input/images && \
    mv /tmp/algorithm/devel src/devel && \
    mv /tmp/algorithm/experiments src/experiments && \
    rm -rf /tmp/algorithm

# Install additional spider baseline dependencies
RUN pip3 install --no-cache-dir git+https://github.com/DIAGNijmegen/Tiger.git@stable

# TODO Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_spider_baseline" \
#    && git fetch https://github.com/MHubAI/models.git main

# Add two static configuration json files required by the SPIDER baseline algorithm
RUN echo '{"filters": 50, "traversal_direction": "up"}' > /app/src/experiments/SPIDER-Baseline/arguments.json
RUN echo '[{\
    "identifier": "input_img",\
    "subset": "testing",\
    "filetype": "mha",\
    "slice-order-reversed": false,\
    "rescale-intercept": 0,\
    "modality": "MRI",\
    "ignore_slices_top": 0,\
    "ignore_slices_bottom": 0\
}]' > /app/src/datasets/spider_input/metadata.json

# Setup required environment variables for SPIDER algorithm
ENV PYTHONPATH="/app:/app/src/devel"
ENV VERSEG_BASEDIR="/app/src/"

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_spider_baseline/config/default.yml"]
