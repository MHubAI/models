# Specify the base image for the environment
FROM ubuntu:20.04

# Authors of the image
LABEL authors="lnuernberg@bwh.harvard.edu,dbontempi@bwh.harvard.edu"

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a working directory and set it as the working directory
# Also create directories for input and output data (mounting points) in the same RUN to avoid creating intermediate layers
RUN mkdir /app /app/data /app/data/input_data /app/data/output_data /app/models
WORKDIR /app

# Install system utilities and useful packages
RUN apt update && apt install -y --no-install-recommends \  
  wget \
  curl \
  jq \
  gcc \
  unzip \
  sudo \
  git \
  python3 \
  python3-pip

# Install core MHubIO modules requirements (python3-openslide and libvips-dev are dependencies of panimg)
RUN apt update && apt install -y --no-install-recommends \
  libopenslide-dev \
  libvips-dev \
  plastimatch \
  dcm2niix \
  ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/* 

# Install general python utilities (specify version if necessary)
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir \
  typing-extensions \
  Pillow==9.5.0 \
  h5py \
  numpy \
  pandas \ 
  panimg \
  pydicom \
  pydicom-seg \
  highdicom \
  rt_utils \
  PyYAML \
  pyplastimatch \
  SimpleITK==2.2.1 \
  thedicomsort \
  colorspacious \
  jsonschema==4.21.1

# Install mhubio framework from git, pulls the utils scripts from GitHub
RUN pip3 install git+https://github.com/MHubAI/mhubio.git \
  && pip3 install git+https://github.com/MHubAI/segdb.git \
  && git init \
  && git sparse-checkout set "base/buildutils" "base/bin" "base/configs" \
  && git fetch https://github.com/MHubAI/models.git main \
  && git merge FETCH_HEAD \
  && mv base/buildutils . \
  && chmod +x base/bin/* \
  && cp base/bin/* /usr/bin/ \
  && cp -r base/configs /app/models/utility \
  && rm -r base \
  && rm -r .git

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
