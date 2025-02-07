FROM mhubai/base:latest

# Install pyradiomics 3.0.1 into a separate python 3.8 environment
# NOTE: pyradiomics 3.1.0 does not provice a cli entrypoint
# NOTE: pyradiomics 3.0.1 somehow doesn't correctly pick up the numpy build dependency. We use --no-build-isolation and manually install setuptools and numpy.
# NOTE: we pin the simpleitk version which is not defined in the pyradiomics pyproject.toml and will differ in the generted report (diagnostic section reports sitk version)
RUN uv venv -p 3.8 .venv38 \
 && uv pip install -p .venv38/ setuptools==74.1.2 numpy==1.24.4 \
 && uv pip install -p .venv38/ --no-build-isolation pyradiomics==3.0.1 simpleitk==2.4.0

# Import the MHub model definiton
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh pyradiomics ${MHUB_MODELS_REPO}

# Default run script
ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/pyradiomics/config/default.yml"]