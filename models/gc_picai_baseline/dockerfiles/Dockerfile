# Specify the base image for the environment
FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for unpacking model weights)
RUN apt update && apt install -y --no-install-recommends git-lfs && rm -rf /var/lib/apt/lists/*

# Install PICAI baseline algorithm and model weights
#   - Git clone the algorithm repository for v2.1.1 (fixed to v2.1.1 tag)
#   - We remove unnecessary files for a compacter docker layer
#   - Subsequently we remove the .git directory to procuce a compacter docker layer
RUN git clone --depth 1 --branch v2.1.1 https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm.git /opt/algorithm && \
    rm -rf /opt/algorithm/test && \
    rm -rf /opt/algorithm/.git

# Install additional PICAI requirements
RUN pip3 install --no-cache-dir -r /opt/algorithm/requirements.txt

# Extend the nnUNet installation with custom trainers
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /opt/algorithm/nnUNetTrainerV2_focalLoss.py "$SITE_PKG/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py"
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /opt/algorithm/nnUNetTrainerV2_Loss_CE_checkpoints.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_CE_checkpoints.py"
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /opt/algorithm/nnUNetTrainerV2_Loss_FL_and_CE.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_FL_and_CE.py"

# Two code edits to the __init__ method of the algorithm class in process.py to prevent some of its default behavior
# 1. Skip forced error caused by using a different input locations than expected (we don't use the GC dirs)
# 2. Prevent unnecessary folder creation before input directories have been set (we will set the correct directory later)
RUN sed -i "s|file_paths = list(Path(folder).glob(scan_glob_format))|return|g" /opt/algorithm/process.py && \
    sed -i "s|self.cspca_detection_map_path.parent.mkdir(exist_ok=True, parents=True)||g" /opt/algorithm/process.py

# FIXME: temporary fix waiting for the latest base image update
# Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_picai_baseline" \
#    && git fetch https://github.com/MHubAI/models.git main

# Add lobe segmentation code base to python path
ENV PYTHONPATH="/app:/opt/algorithm"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_picai_baseline/config/default.yml"]
