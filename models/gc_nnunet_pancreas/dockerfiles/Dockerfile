# Specify the base image for the environment
FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Clone MHub model (m-gc-nnunet-pancreas branch, fixed to commit 407f1f884f09898bef9a9173e6434d681a50d399) # TODO
#RUN git init \
# && git sparse-checkout set "models/gc_nnunet_pancreas" \
# && git fetch https://github.com/MHubAI/models.git m-gc-nnunet-pancreas \
# && git merge TODO

# Install git-lfs (required for downloading the model weights)
RUN apt update && apt install -y --no-install-recommends \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install the model weights and the algorithm files
#  * Pull algorithm from repo into /opt/algorithm for commit e4f4008c6e18e60a79f693448562a340a9252aa8
#  * Remove .git folder to keep docker layer small
#  * Replace input images path in process.py with an existing folder to avoid errors
RUN git clone --depth 1 https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet.git /opt/algorithm && \
    cd /opt/algorithm && \
    git reset --hard e4f4008c6e18e60a79f693448562a340a9252aa8 && \
    rm -rf /opt/algorithm/.git && \
    sed -i 's/Path("\/input\/images\/")/Path("\/app")/g' /opt/algorithm/process.py

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnUNet and other requirements (should install PyTorch as well...)
RUN pip3 install --no-cache-dir -r /opt/algorithm/requirements.txt

# Extend the nnUNet installation with custom trainers
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /opt/algorithm/nnUNetTrainerV2_Loss_CE_checkpoints.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_CE_checkpoints.py"

# Add algorithm files to python path
ENV PYTHONPATH=/opt/algorithm:/app

# Default run script
CMD ["python3", "/app/models/gc_nnunet_pancreas/scripts/run.py"]