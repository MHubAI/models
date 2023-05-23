# Specify the base image for the environment
FROM mhubai/base:nocuda

# Specify/override authors label
LABEL authors="s.vandeleemput@radboudumc.nl"

# install required dependencies for lobe segmentation
RUN pip3 install --no-cache-dir \
    dgl===1.0.2 -f https://data.dgl.ai/wheels/repo.html \
    torch===1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

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

# Install MHub code from git via subversion
# TODO this might be better as a copy command, since we could always build the Docker image from /app/models
COPY /xie2020_lobe_segmentation /app/models/xie2020_lobe_segmentation
# TODO replace the SVN checkout with something else as the EOL of the GitHub SVN support is Januari 2014
#RUN svn checkout https://github.com/MHubAI/models/trunk/models/xie2020_lobe_segmentation /app/models/xie2020_lobe_segmentation

# Add lobe segmentation code base to python path
ENV PYTHONPATH="/xie2020_lobe_segmentation:/app"

# Default run script
CMD ["python3", "/app/models/xie2020_lobe_segmentation/scripts/run.py"]
