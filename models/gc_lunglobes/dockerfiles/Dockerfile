FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl,dbontempi@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install required dependencies for lobe segmentation (CUDA-enabled)
RUN pip3 install --no-cache-dir \
    opencv-python \
    torch==2.0.1 torchvision==0.15.2 \
    dgl==1.1.2 -f https://data.dgl.ai/wheels/cu118/repo.html

# SimpleITK downgrade required for legacy Resample::Execute operation
RUN pip3 install --no-cache-dir --force-reinstall SimpleITK==1.2.4 

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_lunglobes ${MHUB_MODELS_REPO}

# Install Xie's pulmonary lobe segmentation algorithm and model weights (release gclobe165 v1.6.5)
RUN git clone --depth 1 --branch v1.6.5 https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation.git src && \
    sed -i 's/from models import CTSUNet/from src.models import CTSUNet/g' src/test.py

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_lunglobes/config/default.yml"]
