FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

ARG MONAI_BUNDLE_DIR='https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/prostate_mri_anatomy_v0.1.0.zip'
ARG MONAI_MODEL_NAME='prostate_mri_anatomy'

# Install nnunet and platipy
RUN python3 -m pip install --upgrade pip && pip3 install --no-cache-dir "monai[ignite]" fire nibabel simpleITK

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh monai_prostate158 ${MHUB_MODELS_REPO}

# Pull weights into the container
ENV WEIGHTS_DIR=/app/models/monai_prostate158/bundle
RUN mkdir -p $WEIGHTS_DIR
RUN python3 -m monai.bundle download "prostate_mri_anatomy" --bundle_dir ${WEIGHTS_DIR}

#define path to bundle root
ENV BUNDLE_ROOT=/app/models/monai_prostate158/bundle/prostate_mri_anatomy

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/monai_prostate158/config/default.yml"]
