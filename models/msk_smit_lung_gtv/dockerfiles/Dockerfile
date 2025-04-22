FROM mhubai/base:latest

# Update authors label
LABEL authors="jiangj1@mskcc.org,aptea@mskcc.org,deasyj@mskcc.org,iyera@mskcc.org,locastre@mskcc.org"

SHELL ["/bin/bash", "-c"]

RUN apt update

ARG MHUB_MODELS_REPO
ENV MODEL_NAME=msk_smit_lung_gtv
RUN buildutils/import_mhub_model.sh msk_smit_lung_gtv ${MHUB_MODELS_REPO}

RUN source /app/models/msk_smit_lung_gtv/src/get_weights.sh

RUN uv venv --python-preference only-managed -p 3.9 .venv39
RUN uv pip install -n -p .venv39  --extra-index-url https://download.pytorch.org/whl/cu116 torch==1.12.1+cu116
RUN uv pip install -n -p .venv39 simpleitk==2.2.1 nibabel==4.0.2 monai==0.8.0 timm==0.6.11 ml-collections==0.1.1 einops==0.8.1 scikit-image==0.19.3 Cmake imagecodecs monai==0.8.0 torchaudio==0.12.1 pytorch-ignite==0.4.8
RUN uv pip install -n -p .venv39 numpy==1.23.4

ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/msk_smit_lung_gtv/config/default.yml"]
