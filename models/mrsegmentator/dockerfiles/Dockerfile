FROM mhubai/base:latest

# clone mhub implementation
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh mrsegmentator ${MHUB_MODELS_REPO}


# Note: There currently is a dependency issue for nnunet because of batchgenerators. This solves the issue in the description file
RUN git clone https://github.com/MIC-DKFZ/batchgenerators.git /tmp/batchgenerators
# Fix the setup.cfg file - change description-file to description_file
RUN sed -i 's/description-file = README.md/description_file = README.md/' /tmp/batchgenerators/setup.cfg
# Install batchgenerators 
RUN uv pip install -e /tmp/batchgenerators

# Install MRSegmentator
RUN uv pip install mrsegmentator==1.2.3

# Execute mrsegmentator once to download the weights, this will fail to run but download regardless
RUN touch .temp_image.nii.gz
RUN uv run mrsegmentator -i .temp_image.nii.gz; exit 0
RUN rm .temp_image.nii.gz

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/mrsegmentator/config/default.yml"]
