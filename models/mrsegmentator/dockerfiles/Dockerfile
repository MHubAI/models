FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
#ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip

# Install TotalSegmentator
RUN python3 --version
#RUN python3.9 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3.9 -m pip install --no-cache-dir mrsegmentator

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN mrsegmentator -i .temp_image.nii.gz; exit 0

# Import the MHub model definiton
#ARG MHUB_MODELS_REPO
#RUN buildutils/import_mhub_model.sh totalsegmentator ${MHUB_MODELS_REPO}

# Default run script
#ENTRYPOINT ["python3", "-m", "mhubio.run"]
#CMD ["--config", "/app/models/totalsegmentator/config/default.yml"]

CMD ["/bin/bash"]
