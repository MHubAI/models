# Specify the base image for the environment
FROM mhubai/base:cuda11.4

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install panimg to test conversion integration TODO should later be installed with MHub/mhubio
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3-openslide \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir panimg

# install required dependencies for algorithm
RUN pip3 install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

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

# Clone MHub model (m-tiger-lb2 branch, fixed to commit TODO)
#RUN git init \
# && git sparse-checkout set "models/tiger_lb2" \
# && git fetch https://github.com/MHubAI/models.git m-tiger-lb2 \
# && git merge TODO

# Add algorithm code base and model code bases to python path
# NOTE yolov5 and /app both have a models directory in their folders which causes import conflicts
# to solve this we add /app/models instead of /app to the python path
ENV PYTHONPATH="/vuno:/app/models:/usr/local/lib/python3.8/dist-packages/yolov5/"

# Default run script
CMD ["python3", "/app/models/tiger_lb2/scripts/run.py"]
