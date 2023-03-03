# Specify the base image for the environment
FROM mhubai/base:cuda11.4

# Install TotalSegmentator
RUN pip3 install --no-cache-dir TotalSegmentator 

# Clone MHub model
RUN svn checkout https://github.com/MHubAI/models/trunk/models/totalsegmentator /app/models/totalsegmentator

# Download weights using the script in utils
# Usage: utils/download_weights.sh <weights_dir> <url1> <url2> ...
RUN utils/download_weights.sh \
  /root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/ \
  https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip \
  https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip \
  https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip \
  https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip \
  https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip \
  https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip

# Default run script
CMD ["python3", "/app/models/totalsegmentator/scripts/run.py"]