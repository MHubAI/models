FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install autoPET algorithm
#  - Git clone to tmp directory (main branch, commit 828bbf683da730f6c36be9d53b5a506a0eff3ae4)
#  - Extract relevant files and model weights to /opt/algorithm/
#  - Remove tmp directory
RUN git clone https://github.com/YigePeng/AutoPET_False_Positive_Reduction.git /tmp/algorithm && \
    cd /tmp/algorithm && git reset --hard 828bbf683da730f6c36be9d53b5a506a0eff3ae4 && cd /app && \
    mkdir -p /opt/algorithm && \
    mkdir -p /opt/algorithm/checkpoints && \
    mv /tmp/algorithm/process.py /opt/algorithm/ && \
    unzip /tmp/algorithm/hybrid_weights_1.zip -d /opt/algorithm/checkpoints/hybrid_cnn/ && \
    unzip /tmp/algorithm/hybrid_weights_2.zip -d /opt/algorithm/checkpoints/hybrid_cnn/ && \
    unzip /tmp/algorithm/weights.zip -d /opt/algorithm/checkpoints/nnUNet/ && \
    rm -rf /tmp/algorithm

# Install nnunet as a dependency
RUN pip3 install --no-cache-dir nnunet==1.7.1

# nnUNet specific setup
RUN mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task504_Total_PET_Lesion_Only/result

# Configure nnUNet paths
ENV nnUNet_raw_data_base="/opt/algorithm/nnUNet_raw_data_base"
ENV RESULTS_FOLDER="/opt/algorithm/checkpoints"

# Import the MHub model definition
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_autopet_fpr ${MHUB_MODELS_REPO}

# Add /opt/algorithm to the PYTHONPATH to be able to import the processor code
ENV PYTHONPATH "/app:/opt/algorithm"

# Configure main entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_autopet_fpr/config/default.yml"]
