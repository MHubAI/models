# Specify the base image for the environment
FROM mhubai/base:cuda11.4

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet (for heart segmentation) and casust dependencies (for cardiac substructure segmentation)
RUN pip3 install --no-cache-dir \
  nnunet \
  torch \
  torchvision \
  torchio

# Install the latest mhubio and segdb version (not required if base images are up tp date)
#RUN pip3 uninstall -y mhubio segdb \
# && pip3 install git+https://github.com/MHubAI/mhubio.git \
# && pip3 install git+https://github.com/MHubAI/segdb.git

# Clone MHub model (m-casust branch, fixed to commit be83ca516729ce2ca9b578d53e11cf5745f58164)
RUN git init \
 && git sparse-checkout set "models/casust" \
 && git fetch https://github.com/MHubAI/models.git m-casust \
 && git merge be83ca516729ce2ca9b578d53e11cf5745f58164

# Clone the casust model
RUN git clone https://github.com/LennyN95/CaSuSt /app/models/casust/src

# pull weights for casust so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/app/models/casust/src/weights/"
ENV WEIGHTS_URL="https://zenodo.org/record/7836696/files/casust_weights_7roi.zip"
ENV WEIGHTS_FN="casust_weights_7roi.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.platipy/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip"
ENV WEIGHTS_FN="Task400_OPEN_HEART_3d_lowres.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
CMD ["python3", "-m", "mhub.run", "--config", "/app/models/casust/config/config.yml"]