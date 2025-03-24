FROM mhubai/base:latest

# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mrsegmentator ${MHUB_MODELS_REPO}


# Install Python venv package and create a virtual environment
# Install Python venv package and create a virtual environment with a different name
# Force setuptools version 77.0.0 before installing MRSegmentator
RUN uv pip uninstall setuptools
RUN uv pip install setuptools==77.0.3

# Install MRSegmentator using uv
RUN uv pip install mrsegmentator==1.2.2

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN uv run -p .venv_mrseg mrsegmentator -i .temp_image.nii.gz; exit 0
RUN rm .temp_image.nii.gz

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/mrsegmentator/config/default.yml"]
