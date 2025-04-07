FROM mhubai/base:latest

# Update authors label
LABEL authors="aptea@mskcc.org,deasyj@mskcc.org,iyera@mskcc.org,locastre@mskcc.org"

RUN apt update && apt install -y python3.9 python3.9-distutils python3-pip
RUN pip install virtualenv

ARG MHUB_MODELS_REPO
#ENV MHUB_MODELS_REPO=https://github.com/locastre/models.git
ENV MODEL_NAME=msk_smit_lung_gtv
RUN buildutils/import_mhub_model.sh ${MODEL_NAME} ${MHUB_MODELS_REPO}

#ENV WORK_DIR=/app/models/msk_smit_lung_gtv/src
ENV WORK_DIR=/app


WORKDIR ${WORK_DIR}/${MODEL_NAME}/src
ENV WEIGHTS_URL=https://mskcc.box.com/shared/static/sf7jic4m2dk67413cipbbq6hddvhpj61.gz
RUN wget ${WEIGHTS_URL} -O weights.tar.gz && tar xvf weights.tar.gz && rm weights.tar.gz

WORKDIR ${WORK_DIR}
RUN virtualenv -p /usr/bin/python3.9 unetr && source unetr/bin/activate
RUN pip install -r ${WORK_DIR}/${MODEL_NAME}/src/requirements.txt

ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/msk_smit_lung_gtv/config/default.yml"]
