FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet and platipy
RUN pip3 install --no-cache-dir \
  nnunet==1.7.1 \
  torch==2.4.1

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh bamf_nnunet_mr_prostate ${MHUB_MODELS_REPO}

# Pull weights into the container
ENV WEIGHTS_DIR=/root/.nnunet/nnUNet_models/nnUNet/
RUN mkdir -p $WEIGHTS_DIR
ENV WEIGHTS_FN=Task788_Prostate.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8290093/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/bamf_nnunet_mr_prostate/config/default.yml"]
