FROM mhubai/base:latest

# Clone the main branch of MHubAI/models
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mhubio_collection_test ${MHUB_MODELS_REPO}

# Setup additional mhubio collections
RUN buildutils/import_mhubio_collection.sh test_collection

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/mhubio_collection_test/config/default.yml"]
