FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install dependencies
RUN pip3 install --no-cache-dir \
    nnunet==1.7.1 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchio==0.19.1 \
 && pip3 install --no-cache-dir --force-reinstall \
    "dicom2nifti<2.6" \
    numpy==1.26.4 \
    pydicom==2.4.4

# pull weights for platipy's nnU-Net so that the user doesn't need to every time a container is run
ENV WEIGHTS_DIR="/root/.platipy/nnUNet_models/nnUNet/"
ENV WEIGHTS_URL="https://zenodo.org/record/6585664/files/Task400_OPEN_HEART_3d_lowres.zip"
ENV WEIGHTS_FN="Task400_OPEN_HEART_3d_lowres.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL}
RUN unzip ${WEIGHTS_DIR}${WEIGHTS_FN} -d ${WEIGHTS_DIR}
RUN rm ${WEIGHTS_DIR}${WEIGHTS_FN}

# specify nnunet specific environment variables
ENV WEIGHTS_FOLDER=$WEIGHTS_DIR

# Import the MHub model definition
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh deepcac2 ${MHUB_MODELS_REPO}

# download the DeepCAC2 model weights
ENV DEEPCAC2_WEIGHTS_DIR="/app/models/deepcac2/src/weights/"
ENV DEEPCAC2_WEIGHTS_FN="vivid-haze-6_model.pt"
ENV DEEPCAC2_WEIGHTS_URL="https://www.dropbox.com/scl/fi/4rodcp9y2vula8loh1v6r/vivid-haze-6_model.pt?rlkey=7j3cyaltvdpthukchn5xbh01d&dl=1"
RUN mkdir -p ${DEEPCAC2_WEIGHTS_DIR} \
 && wget -o /dev/stdout -O ${DEEPCAC2_WEIGHTS_DIR}${DEEPCAC2_WEIGHTS_FN} ${DEEPCAC2_WEIGHTS_URL}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--workflow", "default"]
