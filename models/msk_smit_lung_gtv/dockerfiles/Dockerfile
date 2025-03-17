FROM mhubai/base:latest

# Update authors label
LABEL authors="aptea@mskcc.org,deasyj@mskcc.org,iyera@mskcc.org,locastre@mskcc.org"

RUN apt update

ARG MHUB_MODELS_REPO
#ENV MHUB_MODELS_REPO=https://github.com/locastre/models.git
RUN buildutils/import_mhub_model.sh msk_smit_lung_gtv ${MHUB_MODELS_REPO}

#ENV WORK_DIR=/app/models/msk_smit_lung_gtv/src
ENV WORK_DIR=/app

WORKDIR ${WORK_DIR}/msk_smit_lung_gtv/src
ENV WEIGHTS_URL=https://mskcc.box.com/shared/static/sf7jic4m2dk67413cipbbq6hddvhpj61.gz
ENV CONDA_URL=https://mskcc.box.com/shared/static/d580gfjzzmt26v8klwp8pivb6wafomag.gz
RUN wget ${WEIGHTS_URL} -O weights.tar.gz && tar xvf weights.tar.gz && rm weights.tar.gz
RUN mkdir conda-pack && chmod -R 777 conda-pack
RUN cd conda-pack && wget ${CONDA_URL} -O conda.tar.gz && tar xvf conda.tar.gz && rm conda.tar.gz


ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/msk_smit_lung_gtv/config/default.yml"]
