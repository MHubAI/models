FROM mhubai/base:latest

# Update authors label
LABEL authors="aptea@mskcc.org,deasyj@mskcc.org,iyera@mskcc.org,locastre@mskcc.org"

RUN apt update
RUN mkdir -p /app/models/msk_smit_lung_gtv
RUN cd /app/models/msk_smit_lung_gtv && git clone https://github.com/cerr/model_installer.git && cd model_installer && source installer.sh -m 4 -d /app/models/msk_smit_lung_gtv -p C
RUN chmod -R 755 /app/models/msk_smit_lung_gtv/CT_Lung_SMIT

ENV PYTHONPATH="/app:/app/models/msk_smit_lung_gtv/CT_Lung_SMIT/conda-pack"

RUN source /app/models/msk_smit_lung_gtv/CT_Lung_SMIT/conda-pack/bin/activate

ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/msk_smit_lung_gtv/config/default.yml"]
