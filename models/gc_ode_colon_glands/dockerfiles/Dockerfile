FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install ode segmentation algorithm
#  - Git shallow clone to tmp directory
#  - Extract relevant files to /opt/algorithm/
#  - Remove tmp directory
RUN git clone --depth 1 https://github.com/DIAGNijmegen/neural-odes-segmentation.git /tmp/algorithm && \
    mkdir -p /opt/algorithm && \
    mv /tmp/algorithm/*.py /opt/algorithm/ && \
    mv /tmp/algorithm/best_border_unode_paper.pt /opt/algorithm/ && \
    mv /opt/algorithm/models.py /opt/algorithm/ode_models.py && \
    rm -rf /tmp/algorithm

# Install additional dependencies
RUN pip3 install --no-cache-dir torchdiffeq==0.2.3

# TODO Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_ode_colon_glands" \
#    && git fetch https://github.com/MHubAI/models.git main

# Add /opt/algorithm to the PYTHONPATH to be able to import the processor code
ENV PYTHONPATH "/app:/opt/algorithm"

# Configure main entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_ode_colon_glands/config/default.yml"]
