FROM mhubai/base:latest

# Authors of the image
LABEL authors="lnuernberg@bwh.harvard.edu"

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# isntall additional system dependencies
RUN apt update && apt install -y dcm2niix

# install additional python dependencies
RUN pip3 install --no-cache-dir \
    nnunet \
    nibabel

# pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.nnunet/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://www.dropbox.com/s/igpwt45v6hlquxp/Task005_Prostate.zip"
ENV WEIGHTS_FN="Task005_Prostate.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh nnunet_prostate_zonal_task05 ${MHUB_MODELS_REPO}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/nnunet_prostate_zonal_task05/config/default.yml"]