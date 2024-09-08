FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install ASAP 2.1
RUN apt-get update && \
    apt-get -y install curl libpython3.8-dev && \
    curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb" && \
    dpkg --install ASAP-2.1-py38-Ubuntu2004.deb || true && \
    apt-get -f install --fix-missing --fix-broken --assume-yes && \
    ldconfig -v && \
    apt-get clean && \
    rm ASAP-2.1-py38-Ubuntu2004.deb

# Install tiger LB2 algorithm
#   - Clone tiger LB2 codebase (master branch, fixed to version 0.1.0)
#   - Subsequently we remove the .git directory to procuce a compacter docker layer
RUN git clone --depth 1 --branch 0.1.0 https://github.com/DIAGNijmegen/tiger_vuno_algorithm /vuno && \
    rm -rf /vuno/.git

# Setup and install algorithm pipenv environment
#   1. Ensure we configure a new empty virtual environment for Python 3.8
#   2. Link ASAP libraries into our environment
#   3. Install required torch and torchvision dependencies
#   4. Install tiger LB2 dependencies from requirements.txt
#   5. Upgrade version of numpy and numba to function correctly with ASAP
RUN uv venv --python 3.8 .venv38 \
 && echo "/opt/ASAP/bin" > /app/.venv38/lib/python3.8/site-packages/asap.pth \
 && uv pip install -n -p .venv38 -f https://download.pytorch.org/whl/torch_stable.html \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
 && uv pip install -n -p .venv38 -r /vuno/requirements.txt \
 && uv pip install -n -p .venv38 --upgrade numpy==1.24.4 numba==0.58.1

# Download and install model weights file from zenodo
RUN rm -rf /vuno/pretrained_weights && \
    wget https://zenodo.org/record/8112176/files/pretrained_weights.zip -O /vuno/pretrained_weights.zip && \
    unzip /vuno/pretrained_weights.zip -d /vuno && \
    rm /vuno/pretrained_weights.zip

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_tiger_lb2 ${MHUB_MODELS_REPO}

# Add model and algorithm code bases to python path
ENV PYTHONPATH="/vuno:/app"

# DEV
COPY models/gc_tiger_lb2/utils /app/models/gc_tiger_lb2/utils

# Set default entrypoint
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_tiger_lb2/config/default.yml"]
