FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl,dbontempi@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install required dependencies for lobe segmentation (CUDA-enabled)
RUN pip3 install --no-cache-dir \
    opencv-python \
    torch torchvision \
    dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# SimpleITK downgrade required for legacy Resample::Execute operation
RUN pip3 install --no-cache-dir --force-reinstall SimpleITK==1.2.4 

# FIXME: temporary fix waiting for the latest base image update
# Clone the main branch of MHubAI/models
RUN git stash \
    && git fetch https://github.com/MHubAI/models.git main \
    && git merge FETCH_HEAD \
    && git sparse-checkout set "models/gc_lunglobes" \
    && git fetch https://github.com/MHubAI/models.git main

# Install Xie's pulmonary lobe segmentation algorithm and model weights (main branch commit at 2023/09/13)
RUN git clone https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation.git src && \
    cd src && git reset --hard 5a64b70504d46c042c30851a69cec370f1202e67 && cd /app && \
    sed -i 's/from models import CTSUNet/from src.models import CTSUNet/g' src/test.py

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_lunglobes/config/default.yml"]
