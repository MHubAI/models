# Specify the base image for the environment
FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install system dependencies for platipy
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install nnunet and platipy
RUN pip3 install --no-cache-dir \
  nnunet \
  platipy[cardiac]==0.7.0

# Clone the main branch of MHubAI/models
RUN git stash \
 && git sparse-checkout set "models/platipy" \
 && git fetch https://github.com/MHubAI/models.git main \
 && git merge FETCH_HEAD

# pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.platipy/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip"
ENV WEIGHTS_FN="Task400_OPEN_HEART_3d_lowres.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# pull weights for platipy's cardiac atlas so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.platipy/cardiac/"
ENV WEIGHTS_URL="https://zenodo.org/record/6592437/files/open_atlas.zip"
ENV WEIGHTS_FN="open_atlas.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/platipy/config/default.yml"]
