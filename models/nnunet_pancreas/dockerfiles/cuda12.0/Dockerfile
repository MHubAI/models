# Specify the base image for the environment
FROM mhubai/base:cuda12.0

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet
RUN pip3 install --no-cache-dir \
  nnunet 

# Clone MHub model (m-nnunet-pancreas branch, fixed to commit 407f1f884f09898bef9a9173e6434d681a50d399)
RUN git init \
 && git sparse-checkout set "models/nnunet_pancreas" \
 && git fetch https://github.com/MHubAI/models.git m-nnunet-pancreas \
 && git merge 407f1f884f09898bef9a9173e6434d681a50d399

# pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.nnunet/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://zenodo.org/record/4485926/files/Task007_Pancreas.zip"
ENV WEIGHTS_FN="Task007_Pancreas.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
CMD ["python3", "-m", "mhubio.run", "--config", "/app/models/nnunet_pancreas/config/config.yml"]