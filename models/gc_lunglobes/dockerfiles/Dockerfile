FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl,dbontempi@bwh.harvard.edu,lnuernberg@bwh.harvard.edu"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# create new virtual environment
RUN uv venv --python-preference only-managed -p 3.8 .venv38

# Install required dependencies for lobe segmentation (CUDA-enabled)
#  SimpleITK==1.2.4 is required for legacy Resample::Execute operation
RUN uv pip install -n -p .venv38 \
    pydicom==2.4.4 packaging==24.1 psutil==6.0.0 \
    opencv-python==4.10.0.84 \
    torch==2.0.1 torchvision==0.15.2 \
    SimpleITK==1.2.4                        

# Install dgl (CUDA-enabled)
# NOTE: uv pip install -f option doesn't work as intended
RUN uv pip install -n -p .venv38 pip \
 && uv run -p .venv38 pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/cu118/repo.html

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_lunglobes ${MHUB_MODELS_REPO}

# Install Xie's pulmonary lobe segmentation algorithm and model weights (release gclobe165 v1.6.5)
RUN git clone --depth 1 --branch v1.6.5 https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation.git src && \
    sed -i 's/from models import CTSUNet/from src.models import CTSUNet/g' src/test.py

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_lunglobes/config/default.yml"]
