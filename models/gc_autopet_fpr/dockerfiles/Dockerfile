FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install autoPET algorithm
#  - Git shallow clone to tmp directory
#  - Extract relevant files and model weights to /opt/algorithm/
#  - Remove tmp directory
RUN git clone --depth 1 https://github.com/YigePeng/AutoPET_False_Positive_Reduction.git /tmp/algorithm && \
    mkdir -p /opt/algorithm && \
    mkdir -p /opt/algorithm/checkpoints && \
    mv /tmp/algorithm/process.py /opt/algorithm/ && \
    unzip /tmp/algorithm/hybrid_weights_1.zip -d /opt/algorithm/checkpoints/hybrid_cnn/ && \
    unzip /tmp/algorithm/hybrid_weights_2.zip -d /opt/algorithm/checkpoints/hybrid_cnn/ && \
    unzip /tmp/algorithm/weights.zip -d /opt/algorithm/checkpoints/nnUNet/ && \
    rm -rf /tmp/algorithm

# nnUNet specific setup TODO check if required else remove...
# RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/imagesTs
#RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/result

# Configure nnUNet paths
ENV nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base"
ENV RESULTS_FOLDER="/opt/algorithm/checkpoints"

# TODO Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_autopet_fpr" \
#    && git fetch https://github.com/MHubAI/models.git main

# Add /opt/algorithm to the PYTHONPATH to be able to import the processor code
ENV PYTHONPATH "/app:/opt/algorithm"

# Configure main entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_autopet_fpr/config/default.yml"]