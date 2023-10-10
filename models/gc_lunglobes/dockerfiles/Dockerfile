FROM mhubai/base:latest

# Update authors label
LABEL authors="s.vandeleemput@radboudumc.nl,dbontempi@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install required dependencies for lobe segmentation (CUDA-enabled)
RUN pip3 install --no-cache-dir \
    opencv-python \
    torch==2.0.1 torchvision==0.15.2 \
    dgl==1.1.2 -f https://data.dgl.ai/wheels/cu118/repo.html

# SimpleITK downgrade required for legacy Resample::Execute operation
RUN pip3 install --no-cache-dir --force-reinstall SimpleITK==1.2.4 

# FIXME: temporary fix waiting for the latest base image update
# Clone the main branch of MHubAI/models
RUN git stash \
    && git fetch https://github.com/MHubAI/models.git main \
    && git merge FETCH_HEAD \
    && git sparse-checkout set "models/gc_lunglobes" \
    && git fetch https://github.com/MHubAI/models.git main

# Install Xie's pulmonary lobe segmentation algorithm and model weights
RUN git clone https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation.git src && \
    sed -i 's/from models import CTSUNet/from src.models import CTSUNet/g' src/test.py

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_lunglobes/config/default.yml"]
