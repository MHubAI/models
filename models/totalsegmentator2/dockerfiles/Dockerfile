FROM mhubai/base:latest

# FIXME: set this environment variable as a shortcut to avoid nnunet crashing the build
# by pulling sklearn instead of scikit-learn
# N.B. this is a known issue:
# https://github.com/MIC-DKFZ/nnUNet/issues/1281 
# https://github.com/MIC-DKFZ/nnUNet/pull/1209
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install TotalSegmentator
RUN uv pip install -n totalsegmentator==2.2.1

# Download weights using totalsegmentator utility 
# NOTE: only licence free models are included
RUN totalseg_download_weights -t total \
&& totalseg_download_weights -t total_fast \ 
&& totalseg_download_weights -t total_mr \ 
&& totalseg_download_weights -t total_fast_mr \ 
&& totalseg_download_weights -t lung_vessels \ 
&& totalseg_download_weights -t cerebral_bleed \ 
&& totalseg_download_weights -t hip_implant \ 
&& totalseg_download_weights -t coronary_arteries \ 
&& totalseg_download_weights -t pleural_pericard_effusion \ 
&& totalseg_download_weights -t body \ 
&& totalseg_download_weights -t body_fast

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh totalsegmentator2 ${MHUB_MODELS_REPO}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--workflow", "default"]