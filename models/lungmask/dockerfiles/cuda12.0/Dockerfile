# Specify the base image for the environment
FROM mhubai/base:cuda12.0

# Install LungMask
RUN pip3 install --no-cache-dir \
  git+https://github.com/JoHof/lungmask


# pull weights for the lung segmentation 2D U-Net model
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# pull weights for the lung lobes segmentation 2D U-Net model
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# Default run script
CMD ["python3", "/app/models/lungmask/scripts/run.py"]