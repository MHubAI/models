FROM mhubai/base:latest

# Install dependencies
RUN pip3 install --no-cache-dir \
  pyradiomics==3.0.1

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh pyradiomics ${MHUB_MODELS_REPO}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/pyradiomics/config/default.yml"]