#!/bin/bash

# Script to import the MHub model definition from GitHub.
# provide the name of the model as a parameter.
# Usage: utils/import_mhub_model.sh <model_name> <(repo_url=https://github.com/MHubAI/models.git::main)> <(branch=main)>

# parameters extraction
MODEL_NAME=$1
REPO_AND_BRANCH=${2:-https://github.com/MHubAI/models.git::main}
REPO_URL=$(echo $REPO_AND_BRANCH | awk -F :: '{print $1}')
REPO_BRANCH=$(echo $REPO_AND_BRANCH | awk -F :: '{print $2}')
REPO_BRANCH=${REPO_BRANCH:-$3}
REPO_BRANCH=${REPO_BRANCH:-main}

# printout paramaters (this happens during the docker build...)
echo "Importing model definition from MHub models repository."
echo "├── MODEL NAME ..... ${MODEL_NAME}"
echo "├── REPOSITORY ..... ${REPO_URL}"
echo "└── BRANCH ......... ${REPO_BRANCH}"
echo 

# fail if model name is empty
if [ -z "$MODEL_NAME" ]; then
    echo "Error: no model name provided."
    exit 1
fi

# print a warning that the model definition is not from the 
# the official MHub Models repository and therefore only 
# suitable for development
if [ "$REPO_URL@$REPO_BRANCH" != "https://github.com/MHubAI/models.git@main" ]; then
    echo 
    echo "Warning: the model definition is not from the official MHub Models repository and therefore only suitable for development."
    echo 
fi

# perform a sparse checkout of the model definition folder 
# (models/<model_name>) from the referenced repository and branch
git init
git fetch ${REPO_URL} ${REPO_BRANCH}
git merge FETCH_HEAD
git sparse-checkout set "models/${MODEL_NAME}"
rm -r .git