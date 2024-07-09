FROM mhubai/base:latest AS base
FROM base AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    libbz2-dev \
    libexpat1-dev \
    libxml2-dev \
    wget \
    ninja-build git \
    && rm -rf /var/lib/apt/lists/* 

RUN git config --global url.'https://'.insteadOf 'git://'
# 1. Install ANTs
RUN git clone https://github.com/ANTsX/ANTs.git  /usr/local/src/ants
WORKDIR /build

RUN cmake \
    -GNinja \
    -DBUILD_TESTING=ON \
    -DRUN_LONG_TESTS=OFF \
    -DRUN_SHORT_TESTS=ON \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
    -DCMAKE_INSTALL_PREFIX=/opt/ants \
    /usr/local/src/ants
RUN cmake --build . --parallel
WORKDIR /build/ANTS-build
RUN cmake --install .

ENV PATH="/opt/ants/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/ants/lib:$LD_LIBRARY_PATH"

RUN cmake --build . --target test

# 2. Install fsl
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py \
    && python3 ./fslinstaller.py -d /usr/local/fsl/

FROM base

RUN apt-get update \
    && apt-get install -y --no-install-recommends bc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 3. Copy fsl and ANTs into mhub base image
COPY --from=builder /opt/ants/lib /opt/ants/lib
COPY --from=builder /opt/ants/bin/N4BiasFieldCorrection /opt/ants/bin/N4BiasFieldCorrection
COPY --from=builder /usr/local/fsl /usr/local/fsl

# 4. Install freesurfer for synthstrip and mri_convert
COPY --from=freesurfer/synthstrip /freesurfer/env/bin /freesurfer/env/bin
COPY --from=freesurfer/synthstrip /freesurfer/models /freesurfer/models


ENV PATH="/opt/ants/bin:$PATH:/freesurfer/env/bin:/usr/local/fsl/bin" \
    LD_LIBRARY_PATH="/opt/ants/lib:$LD_LIBRARY_PATH" FREESURFER_HOME="/freesurfer"

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnunet and surfa
RUN pip3 install --no-cache-dir nnunetv2==2.0 surfa==0.6

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh bamf_mr_brain_tumor ${MHUB_MODELS_REPO}

# Pull nnUNet model weights into the container for Dataset009_Breast
ENV WEIGHTS_DIR=/root/.nnunet/nnUNet_models/
RUN mkdir -p $WEIGHTS_DIR
ENV WEIGHTS_FN=Dataset002_BRATS19.zip
ENV WEIGHTS_URL=https://zenodo.org/records/11582627/files/$WEIGHTS_FN
RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/bamf_mr_brain_tumor/config/default.yml"]
