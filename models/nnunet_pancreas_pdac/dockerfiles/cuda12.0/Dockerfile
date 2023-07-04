# syntax=docker/dockerfile:experimental

# Specify the base image for the environment
FROM mhubai/base:cuda12.0

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install panimg to test conversion integration TODO should later be installed with MHub/mhubio
#RUN apt-get update && apt-get install -y --no-install-recommends \
#  python3-openslide \
#  && rm -rf /var/lib/apt/lists/*
#RUN pip3 install panimg

# Clone MHub model (m-nnunet-pancreas branch, fixed to commit 407f1f884f09898bef9a9173e6434d681a50d399) # TODO
#RUN git init \
# && git sparse-checkout set "models/nnunet_pancreas" \
# && git fetch https://github.com/MHubAI/models.git m-nnunet-pancreas \
# && git merge 407f1f884f09898bef9a9173e6434d681a50d399


# Install git-lfs (required for downloading the model weights)
RUN apt update && apt install -y --no-install-recommends \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# TODO remove later ==== Temporariy SSH fix as long as repo is private =======
RUN apt-get update && apt-get install -y --no-install-recommends \
  openssh-client \
  && rm -rf /var/lib/apt/lists/*
# Add github public key to known_hosts for SSH
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
# TODO remove later ===============================


RUN --mount=type=ssh git clone --depth 1 git@github.com:DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet.git /nnunet_pancreas_pdac && \
    cd /nnunet_pancreas_pdac && \
    git reset --hard 117bb4ebf8bc9e90509a468a5d56e0515987b5a7 && \
    rm -rf /nnunet_pancreas_pdac/.git

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet and other requirements
RUN pip3 install --no-cache-dir -r /nnunet_pancreas_pdac/requirements.txt

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=/nnunet_pancreas_pdac/nnunet/results/nnUNet

# Default run script
CMD ["python3", "-m", "mhubio.run", "--config", "/app/models/nnunet_pancreas_pdac/config/config.yml"]