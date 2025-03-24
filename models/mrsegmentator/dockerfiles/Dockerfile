FROM mhubai/base:latest

# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mrsegmentator ${MHUB_MODELS_REPO}


# Install Python venv package and create a virtual environment
RUN apt-get update && \
    apt-get install -y python3-venv && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install mrsegmentator blosc2 acvl-utils==0.2

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN uv run mrsegmentator -i .temp_image.nii.gz; exit 0
RUN rm .temp_image.nii.gz

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/mrsegmentator/config/default.yml"]
