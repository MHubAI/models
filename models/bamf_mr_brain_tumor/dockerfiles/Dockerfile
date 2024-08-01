FROM mhubai/base:latest

ENV DEBIAN_FRONTEND "noninteractive"
ENV LANG            "en_GB.UTF-8"

# 1. Install gcc-14 and build ants
RUN apt-get update && apt-get install -y cmake make ninja-build git bzip2 flex manpages-dev g++ wget unzip file
RUN wget https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-11.4.0.zip \
    && unzip gcc-11.4.0.zip && cd gcc-releases-gcc-11.4.0 && ./contrib/download_prerequisites \
    && mkdir /app/gcc-build && cd /app/gcc-build \
    && ../gcc-releases-gcc-11.4.0/configure -v --target=x86_64-linux-gnu --prefix=/usr/local/gcc-11.4.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib --program-suffix=-11.4 \
    && make -j$(nproc) && make install-strip \
    && update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-11.4.0/bin/gcc-11.4 11 --slave /usr/bin/g++ g++ /usr/local/gcc-11.4.0/bin/g++-11.4 \
    && rm -rf /app/gcc-*

ENV PATH=/usr/local/gcc-11.4.0/bin:$PATH
ENV LD_LIBRARY_PATH="/usr/local/gcc-11.4.0/lib64:$LD_LIBRARY_PATH"

RUN git clone https://github.com/ANTsX/ANTs.git /usr/local/src/ants \
    && mkdir /app/build && cd /app/build && cmake -DBUILD_TESTING=ON \
    -DRUN_LONG_TESTS=OFF \
    -DRUN_SHORT_TESTS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/ants \
    /usr/local/src/ants \
    && cmake --build . --parallel \
    && cd /app/build/ANTS-build && cmake --build . --target test \
    && cmake --install . && rm -rf /app/build

# 2. Install mri_convert
ENV FREESURFER_HOME="/freesurfer"
RUN git clone https://github.com/freesurfer/freesurfer.git && cd freesurfer/mri_synthstrip \
    git+https://github.com/freesurfer/surfa.git@0d83332351083b33c4da221e9d10a63a93ae7f52 \
    && mkdir -p $FREESURFER_HOME/models/ \
    && git remote add datasrc https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git \
    && apt-get update -y && apt-get install -y git-annex \
    && git fetch datasrc && git-annex get . \
    && cp mri_synthstrip $FREESURFER_HOME \
    && cp synthstrip.*.pt $FREESURFER_HOME/models/

# 3. Install fsl
RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python3 ./fslinstaller.py -V 6.0.7.11 -d /usr/local/fsl/ && rm fslinstaller.py

ENV PATH="/opt/ants/bin:$PATH:/$FREESURFER_HOME:/usr/local/fsl/bin" \
LD_LIBRARY_PATH="/opt/ants/lib:$LD_LIBRARY_PATH"

# 3. Install nnunet
# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# 4. Install nnunet and surfa
RUN pip3 install --no-cache-dir torch surfa==0.6
RUN apt-get update && apt-get install -y python3.9 python3-pip && apt-get clean  && rm -rf /var/lib/apt/lists/*
RUN python3.9 -m pip install nnunetv2==2.4.1
RUN apt-get update && apt-get install -y bc

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh bamf_mr_brain_tumor ${MHUB_MODELS_REPO}

# Pull nnUNet model weights into the container
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
