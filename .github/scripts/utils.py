from typing import List, Union, TypedDict, Optional
from enum import Enum
import requests, os
import json
import jsonschema

# NOTE: all file path operations are relative to the repository root.

# references for documentation
class DocuRef(Enum):
    MODEL_FOLDER_STRUCTURE = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/model_folder_structure.md"
    DOCKERFILE = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/the_mhub_dockerfile.md"
    CONFIG = "https://github.com/MHubAI/documentation/blob/main/documentation/mhubio/the_mhubio_config_file.md"
    MHUBIO_MODULES = "https://github.com/MHubAI/documentation/blob/main/documentation/mhubio/mhubio_modules.md"
    MODEL_META_JSON = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/model_json.md"

class MhubIOCollection(TypedDict):
    name: str
    repo: Optional[str]
    branch: Optional[str]

class MHubComplianceError(Exception):
    """Raised when a model is not compliant with MHub standards"""

    def __init__(self, message: str, docu_ref: Union[DocuRef, List[DocuRef]]):
        if isinstance(docu_ref, list):
            msg = f"{message} (see {', '.join([d.value for d in docu_ref])})"
        else:
            msg = f"{message} ( see {docu_ref.value} for more information)"

        super().__init__(msg)

def get_modified_files_from_PR(prid, repo = 'models') -> List[str]:
    
    # GitHub API URL to list files modified in the PR
    api_url = f"https://api.github.com/repos/MHubAI/{repo}/pulls/{prid}/files"

    # Send a GET request to the GitHub API
    response = requests.get(api_url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch modified files: {response.status_code}")

    # Parse the JSON response and extract the file paths
    modified_files = [file["filename"] for file in response.json()]

    # return list of modified files
    return modified_files

def get_modified_models_from_modified_files(modified_files: List[str]) -> List[str]:
    modified_models = []

    # Parse the JSON response and extract the file paths
    for file in modified_files:

        # get the model name (/models/<model_name>/...)
        model_name = file.split("/")[1]
        modified_models.append(model_name)

    # remove duplicates
    modified_models = list(set(modified_models))
        
    return modified_models

def validateModelFolder(base: str, model_name: str):

    model_path = os.path.join(base, model_name)

    # check if the model folder exists
    if not os.path.isdir(model_path):
        raise MHubComplianceError(f"Model folder {model_path} does not exist", DocuRef.MODEL_FOLDER_STRUCTURE)
    
    # check if the model folder contains the following and no additional ressources
    # - /dockerfiles/Dockerfile
    # - /config/default.yml
    # - /utils
    # - /meta.json

    # check if the model folder contains a Dockerfile
    dockerfile_path = os.path.join(model_path, "dockerfiles", "Dockerfile")
    if not os.path.isfile(dockerfile_path):
        raise MHubComplianceError(f"Model folder {model_path} does not contain a Dockerfile", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.DOCKERFILE])
    
    # check if the model folder contains a default config
    config_path = os.path.join(model_path, "config", "default.yml")
    if not os.path.isfile(config_path):
        raise MHubComplianceError(f"Model folder {model_path} does not contain a default workflow configuration", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.CONFIG])
    
    # check if the model folder contains a utils folder
    # NOTE: utils is not mandatory, however, all MHub-IO modules must be inside the utils folder if they exist.
    #       we can check modified files for any *.py and demand they're inside the utils folder.
    #utils_path = os.path.join(model_path, "utils")
    #if not os.path.isdir(utils_path):
    #    raise MHubComplianceError(f"Model folder {model_path} does not contain a utils folder")
    
    # check if the model folder contains a model.json
    model_json_path = os.path.join(model_path, "meta.json")
    if not os.path.isfile(model_json_path):
        raise MHubComplianceError(f"Model folder {model_path} does not contain a meta.json", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.MODEL_META_JSON])
    

def validateModelMetaJson(model_meta_json_file: str):

    # load schema
    with open(os.path.join('.github', 'schemas', 'meta.schema.json'), "r") as f:
        schema = json.load(f)

    # load model meta json
    with open(model_meta_json_file, "r") as f:
        model_meta_json = json.load(f)

    # validate
    try: 
        jsonschema.validate(instance=model_meta_json, schema=schema)
    except jsonschema.ValidationError as e:
        raise MHubComplianceError(f"Model meta json is not compliant with the schema: {e.message}", DocuRef.MODEL_META_JSON)

def validateModelMetaJson_modelName(model_meta_json_file: str, model_name: str):

    # load model meta json
    with open(model_meta_json_file, "r") as f:
        model_meta_json = json.load(f)

    # check that the model name is correct
    if model_meta_json["name"] != model_name:
        raise MHubComplianceError(f"Model name in meta.json does not match model name in folder structure: {model_meta_json['name']} != {model_name}", DocuRef.MODEL_META_JSON)

def validateDockerfile(base: str, model_name: str):
    
    # get dockerfile path
    model_dockerfile = os.path.join(base, model_name, "dockerfiles", "Dockerfile")

    # read dockerfile
    with open(model_dockerfile, "r") as f:
        dockerfile = f.read()

    # split dockerfile into lines
    lines = dockerfile.split("\n")

    # remove empty lines
    lines = [line for line in lines if line.strip() != ""]

    # check that the dockerfile contains only a single FROM command which 
    # is the first line of the file and is `FROM mhubai/base:latest`
    if not lines[0].strip() == "FROM mhubai/base:latest":
        raise MHubComplianceError(f"Dockerfile does not contain the correct FROM command: {lines[0]}", DocuRef.DOCKERFILE)

    # some status variables from parsing the dockerfile
    dockerfile_defines_arg_mhub_models_repo = False
    dockerfile_contains_mhubio_import = False
    dockerfile_mhubio_collections: List[MhubIOCollection] = []

    # check that dockerfile contains no ADD or COPY commands
    # We also don't allow changing the WORKDIR which is set to /app in the base and must be consistent across all models
    #  so no new line is allowed to start with ADD, COPY, WORKDIR, .. 
    for i, line in enumerate(lines):

        # forbidden keywords

        if line.startswith("WORKDIR"):
            raise MHubComplianceError(f"WORKDIR must not be set to any other than `/app` as defined in our base image. {line}", DocuRef.DOCKERFILE)

        if line.startswith("ADD") or line.startswith("COPY"):
            raise MHubComplianceError(f"Dockerfile contains ADD or COPY command: {line}", DocuRef.DOCKERFILE)

        if line.startswith("FROM") and i > 0:
            raise MHubComplianceError(f"Dockerfile contains FROM command not at the beginning of the file: {line}", DocuRef.DOCKERFILE)

        # required keywords & status variables

        if line == "ARG MHUB_MODELS_REPO":
            dockerfile_defines_arg_mhub_models_repo = True

        if line == f"RUN buildutils/import_mhub_model.sh {model_name} ${{MHUB_MODELS_REPO}}":
           dockerfile_contains_mhubio_import = True
           
        if line.startswith("RUN buildutils/import_mhubio_collection.sh"):
            # parse the collection import
            parts = line.split(" ")
            
            # a collection name must be specified, repo url and branch are optional (but not permitted in a later check)
            if len(parts) < 3:
                raise MHubComplianceError(f"Collection import does not contain enough arguments: {line}", DocuRef.DOCKERFILE)
            
            # append to collections list
            dockerfile_mhubio_collections.append({
                "name": parts[2],
                "repo": parts[3] if len(parts) > 3 else None,
                "branch": parts[4] if len(parts) > 4 else None
            })

    # check if the dockerfile contains the required ARG MHUB_MODELS_REPO and model import
    if not dockerfile_defines_arg_mhub_models_repo:
        raise MHubComplianceError(f"Dockerfile does not define 'ARG MHUB_MODELS_REPO'", DocuRef.DOCKERFILE)
    
    if not dockerfile_contains_mhubio_import:
        raise MHubComplianceError(f"Dockerfile does not contain the required mhubio import command: 'RUN buildutils/import_mhub_model.sh {model_name} ${{MHUB_MODELS_REPO}}'.", DocuRef.DOCKERFILE)
    
    # in case the model contains any mhubio-collection imports, check that they all are imported by theri name since custom imports are not allowed
    #  and also check that the model name is registered under base/collections.json
    if len(dockerfile_mhubio_collections) > 0:

        # load all registered officially suported collections
        with open(os.path.join('base', 'collections.json'), "r") as f:
            mhubio_collections = json.load(f)
            
        # check that all collections are registered
        for collection in dockerfile_mhubio_collections:
            if not collection["name"] in mhubio_collections:
                raise MHubComplianceError(f"Collection '{collection['name']}' is unknown. We only allow official collections.", DocuRef.DOCKERFILE)

            # check that the collection is imported by its name
            if collection["repo"] is not None or collection["branch"] is not None:
                raise MHubComplianceError(f"Collection '{collection['name']}' is not imported by its name.", DocuRef.DOCKERFILE)

    # check that the entrypoint of the dockerfile matches
    #  ENTRYPOINT ["mhub.run"]  |  ENTRYPOINT ["python", "-m", "mhubio.run"] (deprecated, no longer allowed)
    if not lines[-2].strip() == 'ENTRYPOINT ["mhub.run"]':
        raise MHubComplianceError(f"Dockerfile does not contain the correct entrypoint: {lines[-2]}", DocuRef.DOCKERFILE)
    
    #  CMD ["--workflow", "default"] | CMD ["--config", "/app/models/$model_name/config/default.yml"]
    if not lines[-1].strip() in ['CMD ["--workflow", "default"]', f'CMD ["--config", "/app/models/{model_name}/config/default.yml"]']:
        raise MHubComplianceError(f"Dockerfile does not contain the correct entrypoint: {lines[-1]}", DocuRef.DOCKERFILE)
    

def get_model_configuration_files(base: str, model_name: str) -> List[str]:

    # get config path
    model_config_dir = os.path.join(base, model_name, "config")

    # get workflow files
    model_workflows = [cf[:-4] for cf in os.listdir(model_config_dir) if cf.endswith(".yml")]

    # return list of configuration files
    return model_workflows