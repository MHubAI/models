FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet (for heart segmentation) and casust dependencies (for cardiac substructure segmentation)
RUN pip3 install --no-cache-dir \
  nnunet==1.7.1 \
  torch \
  torchvision \
  torchio

# Clone the casust model
RUN git clone https://github.com/LennyN95/CaSuSt /app/models/casust/src

# Pull weights for casust so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/app/models/casust/src/weights/"
ENV WEIGHTS_URL="https://zenodo.org/record/7836696/files/casust_weights_7roi.zip"
ENV WEIGHTS_FN="casust_weights_7roi.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# Pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.platipy/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip"
ENV WEIGHTS_FN="Task400_OPEN_HEART_3d_lowres.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# Specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh casust ${MHUB_MODELS_REPO}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/casust/config/default.yml"]