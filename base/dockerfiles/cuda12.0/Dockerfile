# Specify the base image for the environment
FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

# Authors of the image
LABEL authors="lnuernberg@bwh.harvard.edu,dbontempi@bwh.harvard.edu"

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic system utilities and useful packages
# Install common libraries that are needed by a number of models (e.g., nnUNet, Platipy, ...)
# (merge these in a single RUN command to avoid creating intermediate layers)
RUN apt update && apt install -y --no-install-recommends \  
  wget \
  curl \
  jq \
  unzip \
  sudo \
  git \
  subversion \
  python3 \
  python3-pip \
  plastimatch \
  dcm2niix \
  && rm -rf /var/lib/apt/lists/* \
  && apt update && apt install -y ffmpeg libsm6 libxext6 xvfb

# Create a working directory and set it as the working directory
# Also create directories for input and output data (mounting points) in the same RUN to avoid creating intermediate layers
RUN mkdir /app /app/data /app/data/input_data /app/data/output_data
WORKDIR /app

# Install general utilities (specify version if necessary)
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir \
  albumentations \
  h5py \
  nibabel \ 
  numpy \
  opencv-python \
  pandas \ 
  Pillow \
  pydicom \
  PyYAML \
  pyplastimatch \
  scikit-image \
  scikit-learn \
  scipy \
  SimpleITK \
  thedicomsort

# Install mhubio framework from git, pulls the utils scripts from GitHub
RUN pip3 install git+https://github.com/MHubAI/mhubio.git \
  && svn checkout https://github.com/MHubAI/models/trunk/base/utils /app/utils

# Install DCMQI by pulling the latest release from GitHub (via GitHub API)
# Run everything in a single RUN command to avoid creating intermediate layers (and allowing environment variables to be used)
RUN export DCMQI_RELEASE_URL=$(curl -s "https://api.github.com/repos/QIICR/dcmqi/releases/latest" | jq -r '.assets[] | select(.browser_download_url | contains("linux")) | .browser_download_url') \
  && export DCMQI_TAR_FN=$(echo $DCMQI_RELEASE_URL | rev | cut -d "/" -f 1 | rev) \
  && export DCMQI_FN=$(basename $DCMQI_TAR_FN .tar.gz) \
  && export DCMQI_DOWNLOAD_PATH="/app/${DCMQI_TAR_FN}" \
  && export DCMQI_PATH="/app/${DCMQI_FN}" \
  && wget -O $DCMQI_DOWNLOAD_PATH $DCMQI_RELEASE_URL --no-check-certificate \
  && tar -xvf $DCMQI_DOWNLOAD_PATH \
  && sudo mv ${DCMQI_PATH}/bin/* /bin \
  && rm $DCMQI_DOWNLOAD_PATH  \
  && rm -r $DCMQI_PATH

# Set PYTHONPATH to the /app folder
ENV PYTHONPATH="/app"

# FIXME: pass it as a command to the container in Slicer
CMD ["echo", "MHub Ubuntu 20.04 base image. Visit http://mhub.ai to find out more!"]