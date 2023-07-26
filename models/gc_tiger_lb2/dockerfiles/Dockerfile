# Specify the base image for the environment
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
#   - We use a shallow git clone for reduced bandwidth usage
#   - Subsequently we remove the .git directory to procuce a compacter docker layer
RUN git clone --depth 1 https://github.com/vuno/tiger_challenge.git /vuno \
    && rm -rf /vuno/.git

# Install tiger LB2 dependencies
RUN pip3 install --no-cache-dir -r /vuno/requirements.txt

# Reinstall correct version of Numpy to function with ASAP 2.1
RUN pip install --no-cache-dir --force-reinstall numpy==1.22

# Download and install model weights file from zenodo
RUN rm -rf /vuno/pretrained_weights && \
    wget https://zenodo.org/record/8112176/files/pretrained_weights.zip -O /vuno/pretrained_weights.zip && \
    unzip /vuno/pretrained_weights.zip -d /vuno && \
    rm /vuno/pretrained_weights.zip

# Clone MHub model (m-gc-tiger-lb2 branch, fixed to commit TODO)
#RUN git init \
# && git sparse-checkout set "models/gc_tiger_lb2" \
# && git fetch https://github.com/MHubAI/models.git m-gc-tiger-lb2 \
# && git merge TODO

# Add model and algorithm code bases to python path
ENV PYTHONPATH="/vuno:/app"

# Default run script (cannot use configuration yml file due to panimg converter not ...)
CMD ["python3", "/app/models/gc_tiger_lb2/scripts/run.py"]
