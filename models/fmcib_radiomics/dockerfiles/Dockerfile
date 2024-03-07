FROM mhubai/base:latest

LABEL authors="bspai@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# download model weights
RUN wget https://zenodo.org/records/10528450/files/model_weights.torch?download=1 -O /app/model_weights.torch

# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh fmcib_radiomics ${MHUB_MODELS_REPO}


# Install additional pip packages
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir \
  jsonschema==4.21.1

# Install FMCIB package, should install everything else ...
RUN pip3 install foundation-cancer-image-biomarker --pre

ENTRYPOINT ["mhub.run"]
CMD ["--workflow", "default"]
