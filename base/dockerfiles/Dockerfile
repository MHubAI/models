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
RUN mkdir -p /app/data /app/data/input_data /app/data/output_data /app/data/reference_data \
             /app/utility/config /app/xmodules /app/xcollections
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
  tree \
  clang

# Install core MHubIO modules requirements (python3-openslide and libvips-dev are dependencies of panimg)
RUN apt update && apt install -y --no-install-recommends \
  libopenslide-dev \
  libvips-dev \
  plastimatch \
  ffmpeg libsm6 libxext6 \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/* 

# Install uv (download installer, run & remove installer, add to PATH)
ADD https://astral.sh/uv/0.4.4/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.cargo/bin/:$PATH"

# install python3.11 via uv into .venv
RUN uv venv -p 3.11

# Add a link to pip3 for compatibility
# NOTE: model implementations should install dependencies into separate virtual environments and use a cli script.
# RUN uv pip install --no-cache pip && mkdir /mhub-compatibility && ln -s $(uv run which pip) /mhub-compatibility/pip3
# ENV PATH="/mhub-compatibility:$PATH"
RUN uv pip install --no-cache pip \
 && ln -s $(uv run which pip) /usr/bin/pip3

# Install mhub dependencies and tools
RUN uv pip install git+https://github.com/MHubAI/mhubio.git \
  && uv tool install git+https://github.com/LennyN95/medcmp.git

# Install the base image utilities
RUN git init \
  && git sparse-checkout set "base/buildutils" "base/bin" "base/configs" "base/collections.json" \
  && git fetch https://github.com/MHubAI/models.git main \
  && git merge FETCH_HEAD \
  && mv base/buildutils . \
  && chmod +x base/bin/* \
  && cp base/bin/* /usr/bin/ \
  && cp -r base/configs/* /app/utility/config \
  && cp base/collections.json /app/xcollections/collections.json \
  && cp base/collections.json /app/xcollections/collections.json \
  && rm -r base \
  && rm -r .git

# Set PYTHONPATH to the /app folder
ENV PYTHONPATH="/app"

#CMD ["echo", "MHub Ubuntu 20.04 base image. Visit http://mhub.ai to find out more!"]
ENTRYPOINT [ "mhub.run" ]
CMD ["--utility"]
