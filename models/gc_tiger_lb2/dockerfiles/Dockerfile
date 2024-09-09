FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install pipenv (for a custom Python/Pip environment for ASAP-2.1 and the other algorithm requirements)
RUN pip3 install --no-cache-dir pipenv

# Set environment variables for pipenv (installs into /app/.venv)
ENV PIPENV_VENV_IN_PROJECT=1

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
#   1. Ensure we configure a new empty pipenv for Python 3.8
#   2. Link ASAP libraries into our environment
#   3. Install required torch and torchvision dependencies
#   4. Install tiger LB2 dependencies from requirements.txt
#   5. Upgrade version of numpy and numba to function correctly with ASAP
RUN pipenv install --python 3.8 && \
    echo "/opt/ASAP/bin" > /app/.venv/lib/python3.8/site-packages/asap.pth && \
    pipenv run pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    pipenv run pip install --no-cache-dir -r /vuno/requirements.txt && \
    pipenv run pip install --no-cache-dir --upgrade numpy==1.24.4 numba==0.58.1

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

# Set default entrypoint
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/gc_tiger_lb2/config/default.yml"]
