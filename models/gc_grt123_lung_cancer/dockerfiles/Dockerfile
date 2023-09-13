# Specify the base image for the environment
FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# install required dependencies for grt123 algorithm including GPU support
RUN pip3 install --no-cache-dir \
    torch===2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install grt123 algorithm and model weights
#   - Git clone the algorithm repository for v2.0.0 (fixed to v2.0.0 tag commit on 2023/09/13)
#   - We remove unnecessary files for a compacter docker layer
#   - Subsequently we remove the .git directory to procuce a compacter docker layer, but keep the latest commit hash in the HEAD file
RUN git clone --branch v2.0.0 https://github.com/DIAGNijmegen/bodyct-dsb2017-grt123.git /gc_grt123_lung_cancer && \
    cd /gc_grt123_lung_cancer && git reset --hard 9a4ca0415c7fc1d3023a16650bf1cdce86f8bb59 && \
    rm -rf /gc_grt123_lung_cancer/tests && \
    rm -rf /gc_grt123_lung_cancer/training && \
    rm -rf /gc_grt123_lung_cancer/processor && \
    rm -rf /gc_grt123_lung_cancer/images && \
    rm /gc_grt123_lung_cancer/README.md && \
    rm /gc_grt123_lung_cancer/solution-grt123-team.pdf && \
    mv /gc_grt123_lung_cancer/.git/HEAD /gc_grt123_lung_cancer && \
    rm -rf /gc_grt123_lung_cancer/.git/* && \
    mv /gc_grt123_lung_cancer/HEAD /gc_grt123_lung_cancer/.git

# Clone MHub model (m-gc-grt123-lung-cancer branch, fixed to commit TODO)
#RUN git init \
# && git sparse-checkout set "models/grt123_lung_cancer" \
# && git fetch https://github.com/MHubAI/models.git m-gc-grt123-lung-cancer \
# && git merge TODO

# Add lobe segmentation code base to python path
ENV PYTHONPATH="/gc_grt123_lung_cancer:/app"

# Default entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_grt123_lung_cancer/config/default.yml"]
