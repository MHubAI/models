FROM mhubai/base:latest

LABEL authors="bspai@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh bpreg ${MHUB_MODELS_REPO}

# Install additional pip packages
RUN pip3 install --upgrade pip && \
    pip3 install bpreg && \
    pip3 install numpy -U

# download model weights
RUN wget "https://zenodo.org/record/5113483/files/public_bpr_model.zip?download=1" -O "/app/public_inference_model.zip" && \
    unzip "/app/public_inference_model.zip" -d "/app/public_inference_model" && \
    rm "/app/public_inference_model.zip"

ENTRYPOINT ["mhub.run"]
CMD ["--workflow", "default"]
