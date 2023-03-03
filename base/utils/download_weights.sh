#!/bin/bash

# Script to download the weights for the models in the container
# The weights are downloaded to a location specified as an argument, from a (list of) url(s) URL specified as an argument(s)

# Usage: download_weights.sh <weights_dir> <url1> <url2> ...

# Examples:
#   download_weights.sh /root/.platipy/nnUNet_models/nnUNet https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip
#   download_weights.sh /root/.platipy/cardiac https://zenodo.org/record/6592437/files/open_atlas.zip
#
#   download_weights.sh /root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/ \
#     && https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip \
#     && https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip \
#     && ...

WEIGHTS_DIR=$1
WEIGHTS_URL_LIST=${@:2}

# Check that the weights directory exists. If not, create it
if [ ! -d "$WEIGHTS_DIR" ]; then
  mkdir -p "$WEIGHTS_DIR"
fi

# Download the weights specified in the URLs, specifying a directory to download to using --directory-prefix
for url in $WEIGHTS_URL_LIST; do
  
  echo "Downloading weights from ${url}..."
  wget --directory-prefix ${WEIGHTS_DIR} ${url}

  # unzip the downloaded file
  echo "Unzipping weights to ${WEIGHTS_DIR}..."
  unzip ${WEIGHTS_DIR}/*.zip -d ${WEIGHTS_DIR}

  # remove the downloaded zip file
  rm ${WEIGHTS_DIR}/*.zip
done

