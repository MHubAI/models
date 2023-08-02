FROM mhubai/base:latest

# Update authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install stoic2021 baseline algorithm
#  - Git shallow clone to tmp directory
#  - Extract relevant files to /opt/algorithm
#  - Remove tmp directory
RUN git clone --depth 1 https://github.com/luukboulogne/stoic2021-baseline.git /tmp/algorithm && \
    mkdir src && \
    mv /tmp/algorithm/process.py src/process.py && \
    mv /tmp/algorithm/utils.py src/utils.py && \
    mv /tmp/algorithm/algorithm src/algorithm && \
    mv /tmp/algorithm/requirements.txt src/requirements.txt && \
    rm -rf /tmp/algorithm

# Install stoic2021 dependencies
RUN pip3 install --no-cache-dir -r src/requirements.txt

# TODO Clone the main branch of MHubAI/models
#RUN git stash \
#    && git fetch https://github.com/MHubAI/models.git main \
#    && git merge FETCH_HEAD \
#    && git sparse-checkout set "models/gc_stoic_baseline" \
#    && git fetch https://github.com/MHubAI/models.git main

# Set workdir to /app/src to allow algorithm to pick up model weights correctly
WORKDIR "/app/src"

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_stoic_baseline/config/default.yml"]
