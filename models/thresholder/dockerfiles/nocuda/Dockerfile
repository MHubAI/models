# Specify the base image for the environment
FROM mhubai/base:nocuda

# install mhub code from git via subversion
RUN svn checkout https://github.com/MHubAI/models/trunk/models/thresholder /app/models/thresholder

# Default run script
CMD ["python3", "/app/models/thresholder/scripts/run.py"]