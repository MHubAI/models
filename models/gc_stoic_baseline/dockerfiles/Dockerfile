FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install stoic2021 baseline algorithm
#  - Git clone to tmp directory (main branch, commit 88611ed1d49145db547f3db40b6fa917a403c239)
#  - Extract relevant files to src
#  - Remove tmp directory
RUN git clone https://github.com/luukboulogne/stoic2021-baseline.git /tmp/algorithm && \
    cd /tmp/algorithm && git reset --hard 88611ed1d49145db547f3db40b6fa917a403c239 && cd /app && \
    mkdir src && \
    mv /tmp/algorithm/process.py src/process.py && \
    mv /tmp/algorithm/utils.py src/utils.py && \
    mv /tmp/algorithm/algorithm src/algorithm && \
    mv /tmp/algorithm/requirements.txt src/requirements.txt && \
    rm -rf /tmp/algorithm

# Install stoic2021 dependencies
RUN pip3 install --no-cache-dir -r src/requirements.txt

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_stoic_baseline ${MHUB_MODELS_REPO}

# Configure STOIC src path and add it to the python path
ENV STOIC_SRC_PATH=/app/src
ENV PYTHONPATH=/app:${STOIC_SRC_PATH}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_stoic_baseline/config/default.yml"]
