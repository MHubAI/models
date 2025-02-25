FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet and TotalSegmentator
RUN uv pip install p_tqdm==1.4 TotalSegmentator==1.5.7 nnunet==1.6.6 torch==2.2.2 --no-cache-dir

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh bamf_pet_ct_breast_tumor ${MHUB_MODELS_REPO}

# Pull nnUNet model weights into the container for Task777_CT_Nodules
ENV WEIGHTS_DIR=/root/.nnunet/nnUNet_models/nnUNet/
RUN mkdir -p $WEIGHTS_DIR
ENV TASK_NAME=Task762_PET_CT_Breast
ENV WEIGHTS_FN=$TASK_NAME.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8290055/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL} --no-check-certificate
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/bamf_pet_ct_breast_tumor/config/default.yml"]
