FROM mhubai/base:latest

# Install LungMask
RUN pip3 install --no-cache-dir \
    lungmask==0.2.16
  
# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh lungmask ${MHUB_MODELS_REPO}

# pull the weights for the lung segmentation 2D U-Net model
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# pull the weights for the lung lobes segmentation 2D U-Net model 
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/lungmask/config/default.yml"]
