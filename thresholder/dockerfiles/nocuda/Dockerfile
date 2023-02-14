# Specify the base image for the environment
FROM mhubai/base:nocuda

# install mhub code from git via subversion
RUN svn checkout https://github.com/MHubAI/models/trunk/thresholder /app/mhub/thresholder

# Default run script
CMD ["python3", "/app/mhub/thresholder/scripts/run.py"]