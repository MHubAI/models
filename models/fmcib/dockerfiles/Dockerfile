FROM mhubai/base:latest

LABEL authors="bspai@bwh.harvard.edu"



RUN wget https://zenodo.org/records/10528450/files/model_weights.torch?download=1 -O /app/model_weights.torch

RUN mkdir models
RUN mkdir models/fmcib

# Install FMCIB package, should install everything else ...
RUN pip install foundation-cancer-image-biomarker --pre

ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/fmcib/config/default.yml", "--print"]
