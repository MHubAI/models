FROM mhubai/base:latest

# Update Python, nnUNet needs Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip

# Install MRSegmentator
RUN python3.9 -m pip install --no-cache-dir mrsegmentator

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN mrsegmentator -i .temp_image.nii.gz; exit 0

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/mrsegmentator/config/default.yml"]