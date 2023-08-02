FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install node21 baseline algorithm
#  - Git shallow clone to tmp directory
#  - Extract relevant files to /opt/algorithm/
#  - Remove tmp directory
RUN git clone --depth 1 https://github.com/node21challenge/node21_detection_baseline /tmp/algorithm && \
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

# TODO Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_node21_baseline" \
#    && git fetch https://github.com/MHubAI/models.git main

# Configure app /opt/algorithm to be the main workdir and torch hub directory to pick up the model weights correctly
WORKDIR "/opt/algorithm"
ENV TORCH_HOME /opt/algorithm

# Add /opt/algorithm to the PYTHON_PATH to be able to import the processor code
ENV PYTHON_PATH "/app:/opt/algorithm"

#ENTRYPOINT ["python3", "-m", "mhubio.run"]
#CMD ["--config", "/app/models/gc_node21_baseline/config/default.yml"]

# Default run script
CMD ["python3", "/app/models/gc_node21_baseline/scripts/run.py"]
