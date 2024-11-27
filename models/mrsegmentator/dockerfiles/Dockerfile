FROM mhubai/base:latest

# Update Python, nnUNet needs Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip
    
# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mrsegmentator ${MHUB_MODELS_REPO}

# Install MRSegmentator
RUN python3.9 -m pip install --no-cache-dir mrsegmentator==1.2

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN mrsegmentator -i .temp_image.nii.gz; exit 0
RUN rm .temp_image.nii.gz

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--workflow", "default"]