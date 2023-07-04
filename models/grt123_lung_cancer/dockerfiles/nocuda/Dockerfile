# Specify the base image for the environment
FROM mhubai/base:nocuda

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install panimg to test conversion integration TODO should later be installed with MHub/mhubio
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-openslide \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install panimg

# install required dependencies for grt123 algorithm (CPU-only)
RUN pip3 install --no-cache-dir \
    torch===2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install grt123 algorithm and model weights
#   - We use a shallow git clone for reduced bandwidth usage
#   - We remove unnecessary files for a compacter docker layer
#   - Subsequently we remove the .git directory to procuce a compacter docker layer, but keep the latest commit hash in the HEAD file
RUN git clone --depth 1 --branch v2.0.0 https://github.com/DIAGNijmegen/bodyct-dsb2017-grt123.git /grt123_lung_cancer && \
    rm -rf /grt123_lung_cancer/tests && \
    rm -rf /grt123_lung_cancer/training && \
    rm -rf /grt123_lung_cancer/processor && \
    rm -rf /grt123_lung_cancer/images && \
    rm /grt123_lung_cancer/README.md && \
    rm /grt123_lung_cancer/solution-grt123-team.pdf && \
    mv /grt123_lung_cancer/.git/HEAD /grt123_lung_cancer && \
    rm -rf /grt123_lung_cancer/.git/* && \
    mv /grt123_lung_cancer/HEAD /grt123_lung_cancer/.git

# Clone MHub model (m-grt123-lung-cancer branch, fixed to commit TODO)
#RUN git init \
# && git sparse-checkout set "models/grt123_lung_cancer" \
# && git fetch https://github.com/MHubAI/models.git m-grt123-lung-cancer \
# && git merge TODO

# Add lobe segmentation code base to python path
ENV PYTHONPATH="/grt123_lung_cancer:/app"

# Default run script
CMD ["python3", "/app/models/grt123_lung_cancer/scripts/run.py"]
