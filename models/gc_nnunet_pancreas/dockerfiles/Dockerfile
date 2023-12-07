FROM mhubai/base:latest

# Specify/override authors label
LABEL authors="sil.vandeleemput@radboudumc.nl"

# Install PyTorch 2.0.1 (CUDA enabled)
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install git-lfs (required for downloading the model weights)
RUN apt update && \
    apt install -y --no-install-recommends git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Install the model weights and the algorithm files
#  * Pull algorithm from repo into /opt/algorithm (main branch, commit e4f4008c6e18e60a79f693448562a340a9252aa8)
#  * Remove .git folder to keep docker layer small
#  * Replace input images path in process.py with an existing folder to avoid errors
#  * Add specific data types and compression options to output data structures in process.py to reduce generated output footprint
RUN git clone https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet.git /opt/algorithm && \
    cd /opt/algorithm && \
    git reset --hard e4f4008c6e18e60a79f693448562a340a9252aa8 && \
    rm -rf /opt/algorithm/.git && \
    sed -i 's/Path("\/input\/images\/")/Path("\/app")/g' /opt/algorithm/process.py && \
    sed -i 's/pred_2_np  = sitk\.GetArrayFromImage(pred_2_nii)/pred_2_np  = sitk\.GetArrayFromImage(pred_2_nii)\.astype(np\.uint8)/g' /opt/algorithm/process.py && \
    sed -i 's/pm_image = np\.zeros(image_np\.shape)/pm_image = np\.zeros(image_np\.shape, dtype=np\.float32)/g' /opt/algorithm/process.py && \
    sed -i 's/segmentation_np = np\.zeros(image_np\.shape)/segmentation_np = np\.zeros(image_np\.shape, dtype=np\.uint8)/g' /opt/algorithm/process.py && \
    sed -i 's/sitk\.WriteImage(segmentation_image, str(self\.segmentation))/sitk\.WriteImage(segmentation_image, str(self\.segmentation), True)/g' /opt/algorithm/process.py && \
    sed -i 's/sitk\.WriteImage(pred_itk_resampled, str(self\.heatmap))/sitk\.WriteImage(pred_itk_resampled, str(self\.heatmap), True)/g' /opt/algorithm/process.py

# Set this environment variable as a shortcut to avoid nnunet 1.7.0 crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install nnUNet 1.7.0 and other requirements
RUN pip3 install --no-cache-dir -r /opt/algorithm/requirements.txt

# Extend the nnUNet installation with custom trainers
RUN SITE_PKG=`pip3 show nnunet | grep "Location:" | awk '{print $2}'` && \
    mv /opt/algorithm/nnUNetTrainerV2_Loss_CE_checkpoints.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_CE_checkpoints.py"

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh gc_nnunet_pancreas ${MHUB_MODELS_REPO}

# Add algorithm files to python path
ENV PYTHONPATH=/opt/algorithm:/app

# Configure main entrypoint
ENTRYPOINT ["python3", "-m", "mhubio.run"]
CMD ["--config", "/app/models/gc_nnunet_pancreas/config/default.yml"]
