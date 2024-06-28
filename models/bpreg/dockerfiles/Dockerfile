FROM mhubai/base:latest

LABEL authors="bspai@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# # clone mhub implementation
ARG MHUB_MODELS_REPO

# Change later!
RUN mkdir /app/models
# RUN buildutils/import_mhub_model.sh fmcib_radiomics ${MHUB_MODELS_REPO}

# Install additional pip packages
RUN pip3 install --upgrade pip

  # download model weights
RUN pip3 install bpreg
RUN wget "https://zenodo.org/record/5113483/files/public_bpr_model.zip?download=1" -O "/app/public_inference_model.zip"
RUN unzip "/app/public_inference_model.zip" -d "/app/public_inference_model"

# Old numpy version on Bpreg needs to be updated
RUN pip install numpy -U
ENTRYPOINT ["mhub.run"]
CMD ["--workflow", "default"]
