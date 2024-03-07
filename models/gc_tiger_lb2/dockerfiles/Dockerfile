FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# install required dependencies for algorithm
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install ASAP
RUN apt-get update \
    && apt-get -y install curl libpython3.8-dev \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /usr/local/lib/python3.8/dist-packages/asap.pth \
    && rm ASAP-2.1-py38-Ubuntu2004.deb

# Install tiger LB2 algorithm
#   - Clone tiger LB2 codebase (master branch, fixed to commit 720f8dfca4624792c8e57915c4222efec5a0c2d4)
#   - Subsequently we remove the .git directory to procuce a compacter docker layer
#   - We also fix a reproducibility issue with the segmentation prediction by fixing the torch random seed
RUN git clone https://github.com/vuno/tiger_challenge.git /vuno && \
    cd /vuno && git reset --hard 720f8dfca4624792c8e57915c4222efec5a0c2d4 && \
    rm -rf /vuno/.git && \
    sed -i "238i \        torch.manual_seed(0)" /vuno/segmentation/inference.py

# Install tiger LB2 dependencies
RUN pip3 install --no-cache-dir -r /vuno/requirements.txt

# Reinstall correct version of Numpy to function with ASAP 2.1
RUN pip3 install --no-cache-dir --force-reinstall numpy==1.22

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
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_tiger_lb2/config/default.yml"]
