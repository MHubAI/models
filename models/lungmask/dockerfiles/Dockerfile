FROM mhubai/base:latest

# Install LungMask
RUN pip3 install --no-cache-dir \
    git+https://github.com/JoHof/lungmask
  
# FIXME: temporary fix waiting for the latest base image update
# Clone the main branch of MHubAI/models
RUN git stash \
  && git fetch https://github.com/MHubAI/models.git main \
  && git merge FETCH_HEAD \
  && git fetch https://github.com/MHubAI/models.git main \
  && git merge FETCH_HEAD

# pull the weights for the lung segmentation 2D U-Net model
ENV WEIGHTS_DIR="/root/.cache/torch/hub/checkpoints/"
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# pull the weights for the lung lobes segmentation 2D U-Net model 
ENV WEIGHTS_URL="https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}

# Default run script
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/lungmask/config/default.yml"]