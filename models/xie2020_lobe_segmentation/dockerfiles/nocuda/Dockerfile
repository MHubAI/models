# Specify the base image for the environment
FROM mhubai/base:nocuda

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# install required dependencies for lobe segmentation (CPU-only)
RUN pip3 install --no-cache-dir \
    dgl===1.0.2 -f https://data.dgl.ai/wheels/repo.html \
    torch===1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install panimg to test conversion integration TODO should later be installed with MHub/mhubio
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-openslide \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install panimg

# SimpleITK downgrade required for legacy Resample::Execute operation
RUN pip3 install --no-cache-dir --force-reinstall SimpleITK==1.2.4

# Install Xie's pulmonary lobe segmentation algorithm and model weights
#   - We use a shallow git clone for reduced bandwidth usage
#   - Subsequently we remove the .git directory to procuce a compacter docker layer
#   - Finally we rename models.py and references to models_xie.py to avoid naming conflicts with MHub models package
RUN git clone --depth 1 https://github.com/DIAGNijmegen/bodyct-pulmonary-lobe-segmentation.git /xie2020_lobe_segmentation && \
    rm -rf /xie2020_lobe_segmentation/.git && \
    mv /xie2020_lobe_segmentation/models.py /xie2020_lobe_segmentation/models_xie.py && \
    sed -i 's/from models import CTSUNet/from models_xie import CTSUNet/g' /xie2020_lobe_segmentation/test.py

# Clone MHub model (m-lobesegmentation-xie branch, fixed to commit eb679da42dd6b1f93ebbb8c094ec57acab0f14c9)
RUN git init \
 && git sparse-checkout set "models/xie2020_lobe_segmentation" \
 && git fetch https://github.com/MHubAI/models.git m-lobesegmentation-xie \
 && git merge eb679da42dd6b1f93ebbb8c094ec57acab0f14c9

# Add lobe segmentation code base to python path
ENV PYTHONPATH="/xie2020_lobe_segmentation:/app"

# Default run script
CMD ["python3", "/app/models/xie2020_lobe_segmentation/scripts/run.py"]
