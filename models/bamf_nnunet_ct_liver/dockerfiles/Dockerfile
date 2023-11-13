FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet and platipy
RUN pip3 install --no-cache-dir \
  nnunet 

# Clone the main branch of MHubAI/models
RUN git stash \
 && git sparse-checkout set "models/bamf_nnunet_ct_liver" \
 && git fetch https://github.com/MHubAI/models.git main \
 && git merge FETCH_HEAD

# Pull weights into the container
ENV WEIGHTS_DIR=/root/.nnunet/nnUNet_models/nnUNet/
RUN mkdir -p $WEIGHTS_DIR
ENV WEIGHTS_FN=Task773_Liver.zip
ENV WEIGHTS_URL=https://zenodo.org/record/8274976/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/bamf_nnunet_ct_liver/config/default.yml"]
