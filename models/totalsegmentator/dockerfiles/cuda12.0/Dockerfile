# Specify the base image for the environment
FROM mhubai/base:cuda12.0

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install TotalSegmentator
RUN pip3 install --no-cache-dir TotalSegmentator 

# Clone MHub model
RUN svn checkout https://github.com/MHubAI/models/trunk/totalsegmentator /app/mhub/totalsegmentator

# pull weights for TotalSegmentator so that the user doesn't need to every time a container is run
# N.B. - this pulls only the weights for the fast version of Task 256
ENV WEIGHTS_DIR="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/"
ENV WEIGHTS_URL="https://zenodo.org/record/6802052/files/Task256_TotalSegmentator_3mm_1139subj.zip"
ENV WEIGHTS_FN="Task256_TotalSegmentator_3mm_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# Default run script
CMD ["python3", "/app/mhub/totalsegmentator/scripts/run.py"]