FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install TotalSegmentator
RUN pip3 install --no-cache-dir totalsegmentator==1.5.6 

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh totalsegmentator ${MHUB_MODELS_REPO}

# Download weights using the script in utils
# Usage: utils/download_weights.sh <weights_dir> <url1> <url2> ...
RUN buildutils/download_weights.sh \
  /root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/ \
  https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip \
  https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip \
  https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip \
  https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip \
  https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip \
  https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/totalsegmentator/config/default.yml"]