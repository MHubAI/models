FROM mhubai/base:latest AS base
FROM base AS builder

ENV DEBIAN_FRONTEND "noninteractive"
ENV LANG            "en_GB.UTF-8"

# 1 Build and Install ants using g++11.4 into builder
WORKDIR /tmp
RUN apt-get update && apt-get install -y cmake make ninja-build git bzip2 flex manpages-dev g++ wget unzip file build-essential
RUN wget https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-11.4.0.zip \
    && unzip gcc-11.4.0.zip && cd gcc-releases-gcc-11.4.0 && ./contrib/download_prerequisites \
    && mkdir /tmp/gcc-build
WORKDIR /tmp/gcc-build
RUN ../gcc-releases-gcc-11.4.0/configure -v --target=x86_64-linux-gnu --prefix=/usr/local/gcc-11.4.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib --program-suffix=-11.4 \
    && make -j$(nproc) && make install-strip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-11.4.0/bin/gcc-11.4 11 --slave /usr/bin/g++ g++ /usr/local/gcc-11.4.0/bin/g++-11.4 \
    && rm -rf /tmp/gcc-*

ENV PATH=/usr/local/gcc-11.4.0/bin:$PATH
ENV LD_LIBRARY_PATH="/usr/local/gcc-11.4.0/lib64:$LD_LIBRARY_PATH"

RUN git clone https://github.com/ANTsX/ANTs.git /usr/local/src/ants

WORKDIR /tmp/build
RUN cmake \
    -DBUILD_TESTING=ON \
    -DRUN_LONG_TESTS=OFF \
    -DRUN_SHORT_TESTS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/ants \
    /usr/local/src/ants
RUN cmake --build . --parallel
WORKDIR /tmp/build/ANTS-build
RUN cmake --build . --target test
RUN cmake --install .

# 2. Install fsl into builder
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 ./fslinstaller.py -V 6.0.7.11 -d /usr/local/fsl/

# 3. Copy only specific binaries from the builder to reduce memory footprint
FROM base AS final

# Copy fsl binaries and libraries
COPY --from=builder /usr/local/fsl/bin /usr/local/fsl/bin
COPY --from=builder /usr/local/fsl/lib /usr/local/fsl/lib
COPY --from=builder /usr/local/fsl/data /usr/local/fsl/data

# Copy ANTs binary N4BiasFieldCorrection and libraries
COPY --from=builder /opt/ants/bin/N4BiasFieldCorrection /opt/ants/bin/N4BiasFieldCorrection
COPY --from=builder /usr/local/gcc-11.4.0 /usr/local/gcc-11.4.0

# Copy freesurfer for synthstrip
COPY --from=freesurfer/synthstrip:1.5 /freesurfer/env/bin /freesurfer/env/bin
COPY --from=freesurfer/synthstrip:1.5 /freesurfer/models /freesurfer/models

ENV PATH="/opt/ants/bin:$PATH:/freesurfer/env/bin:/usr/local/fsl/bin" \
    LD_LIBRARY_PATH="/usr/local/gcc-11.4.0/lib64:/opt/ants/lib:$LD_LIBRARY_PATH" \
    FREESURFER_HOME="/freesurfer" \
    FSLDIR="/usr/local/fsl"

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN apt-get update && apt-get install -y bc
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

WORKDIR /app
# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/bamf_mr_brain_tumor/config/default.yml"]