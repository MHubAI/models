# Specify the base image for the environment
FROM mhubai/base:cuda12.0

# Install LungMask
RUN pip3 install --no-cache-dir \
  git+https://github.com/JoHof/lungmask


# Install the latest mhubio version
RUN pip3 uninstall -y mhubio \
 && pip3 install git+https://github.com/MHubAI/mhubio.git

# Clone MHub model (m-casust branch, fixed to commit 95fd98c00854c84b95b4845895e67ec98be604e6)
RUN git init \
 && git sparse-checkout set "models/lungmask" \
 && git fetch https://github.com/MHubAI/models.git m-lungmask \
 && git merge 95fd98c00854c84b95b4845895e67ec98be604e6

# pull weights for the lung segmentation 2D U-Net model
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# pull weights for the lung lobes segmentation 2D U-Net model
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# Default run script
CMD ["python3", "/app/models/lungmask/scripts/run.py"]