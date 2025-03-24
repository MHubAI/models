FROM mhubai/base:latest

# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mrsegmentator ${MHUB_MODELS_REPO}


# Install Python venv package and create a virtual environment
# Create a new clean virtual environment for MRSegmentator
RUN uv venv -p 3.11 .venv_mrseg

# Install specific setuptools version in the new environment
RUN uv pip uninstall -p .venv_mrseg setuptools
RUN uv pip install -p .venv_mrseg setuptools==77.0.3

# Install MRSegmentator in the new environment
RUN uv pip install -p .venv_mrseg mrsegmentator==1.2.3

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN uv run -p .venv_mrseg mrsegmentator -i .temp_image.nii.gz; exit 0
RUN rm .temp_image.nii.gz

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/mrsegmentator/config/default.yml"]
