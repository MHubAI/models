#!/bin/bash

# global parameters
ALLOW_UNREGISTERED_COLLECTIONS=${ALLOW_UNREGISTERED_COLLECTIONS:-false}

# this script should fail if any command fails
set -e

# --------------------------------------------------------------------------------------------------- #
# - Functions --------------------------------------------------------------------------------------- #

# this function takes a collection name as argument
#  it will return true if the collection is registered in collections.json (false otherwise)
is_collection_known() {
  local collection_name=$1
  jq -e ".$collection_name" /app/xcollections/collections.json >/dev/null 2>&1
}

# this function takes a collection name as argument
#  it will return the repository of the collection (only for registered collections)
get_collection_repo() {
  local collection_name=$1
  jq -r ".$collection_name.repo" /app/xcollections/collections.json
}

# this function takes a collection name as argument
#  it will return the branch of the collection (only for registered collections) or main if not specified
get_collection_branch() {
  local collection_name=$1

  branch=$(jq -r ".$collection_name.branch" /app/xcollections/collections.json)

  if [ "$branch" == "null" ]; then
    echo "main"
  else
    echo $branch
  fi
}

# this function takes a collection name as argument
#  it will return the version of the collection (only for registered collections)
get_collection_version() {
  local collection_name=$1
  jq -r ".$collection_name.version" /app/xcollections/collections.json
}

# this function takes a github repository as argument
#  it will clone the repository under /app/collections 
import_collection() {
    local collection_name=$1
    local collection_repo=$2

    # clone the mhubio collection repository into a folder named after the collection under /app/collections
    git clone $collection_repo /app/xcollections/$collection_name

    # make collection importable 
    touch /app/xcollections/$collection_name/__init__.py
    touch /app/xcollections/$collection_name/modules/__init__.py
}

# this function will run a collections setup script if it exists
#  the setup script is expected to be in the collection folder under setup/setup.sh
setup_collection() {
    local collection_name=$1
    local collection_folder=/app/xcollections/$collection_name

    # check if the setup script exists
    if [ -f $collection_folder/setup/setup.sh ]; then
        # run the setup script
        bash $collection_folder/setup/setup.sh
    fi
}

# --------------------------------------------------------------------------------------------------- #
# - Main -------------------------------------------------------------------------------------------- #

# run script, take collection name as argument and a collection repository as optional second argument
COLLECTION_NAME=$1

# if the collection is not registered, exit with error
if ! is_collection_known $COLLECTION_NAME || [ ! -z "$2" ]; then

    # if ALLOW_UNREGISTERED_COLLECTIONS is set to true, print a warning and continue, otherwise print an error and exit
    if [ "$ALLOW_UNREGISTERED_COLLECTIONS" == "true" ]; then
        echo ""
        echo "Warning: collection $COLLECTION_NAME is not registered in collections.json and support for external colelctions has been enabled."
        echo ""
    else
        echo ""
        echo "Error: collection $COLLECTION_NAME is not registered in collections.json. You can activate support for external collections by setting ALLOW_UNREGISTERED_COLLECTIONS=true in your Dockerfile."
        echo ""
        exit 1
    fi

    # check if a repository is provided as second argument
    if [ -z "$2" ]; then
        echo "Error: no collection repository provided. For unregistered collections, you need to provide the repository URL as a second argument. You can specify the branch as a third argument or attach it to the repository URL using the :: separator. If no branch is specified, 'main' will be used."
        exit 1
    fi

    # get the repository and branch
    COLLECTION_URL=$(echo $2 | awk -F :: '{print $1}')
    COLLECTION_BRANCH=$(echo $2 | awk -F :: '{print $2}')
    COLLECTION_BRANCH=${COLLECTION_BRANCH:-$3}
    COLLECTION_BRANCH=${COLLECTION_BRANCH:-main}
else
  # get the repository and branch of the collection
  COLLECTION_URL=$(get_collection_repo $COLLECTION_NAME)

  # get the branch of the collection
  COLLECTION_BRANCH=$(get_collection_branch $COLLECTION_NAME)
fi

# printout paramaters (this happens during the docker build...)
echo "Importing additional mhubio collection."
echo "├── COLLECTION NAME .. ${COLLECTION_NAME}"
echo "├── REPOSITORY ....... ${COLLECTION_URL}"
echo "└── BRANCH ........... ${COLLECTION_BRANCH}"
echo 

echo "Importing collection..."

# import the collection
import_collection $COLLECTION_NAME $COLLECTION_URL

# print 
echo "Setting up collection..."

# setup the collection
setup_collection $COLLECTION_NAME

# print (with name reference)
echo "Successfully imported $COLLECTION_NAME collection."
