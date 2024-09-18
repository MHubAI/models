from typing import List, Union, TypedDict, Optional
from enum import Enum
import requests, os
import json
import jsonschema
import zipfile
from colorama import Fore, Style, init as colorama_init
import toml
import io


# NOTE: all file path operations are relative to the repository root.

# initialize colorama
colorama_init(autoreset=True)

# references for documentation
class DocuRef(Enum):
    MODEL_FOLDER_STRUCTURE = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/model_folder_structure.md"
    DOCKERFILE = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/the_mhub_dockerfile.md"
    CONFIG = "https://github.com/MHubAI/documentation/blob/main/documentation/mhubio/the_mhubio_config_file.md"
    MHUBIO_MODULES = "https://github.com/MHubAI/documentation/blob/main/documentation/mhubio/mhubio_modules.md"
    MODEL_META_JSON = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_models/model_json.md"
    MODEL_TEST_PROCEDURE = "https://github.com/MHubAI/documentation/blob/main/documentation/mhub_contribution/testing_phase.md#test-procedure"

class MhubIOCollection(TypedDict):
    name: str
    repo: Optional[str]
    branch: Optional[str]
    
class Message:
    
    def __init__(self, text: str, docu_ref: Union[DocuRef, List[DocuRef], None] = None):
        self._text = text
        
        if docu_ref is None:
            self._message = text
            self._links = []
        elif isinstance(docu_ref, list):
            self._message = f"{text} (see {', '.join([d.value for d in docu_ref])})"
            self._links = [str(d.value) for d in docu_ref]
        else:
            self._message = f"{text} (see {docu_ref.value} for more information)"
            self._links = [str(docu_ref.value)]
        
    @property
    def text(self) -> str:
        return self._text
    
    @property
    def links(self) -> List[str]:
        return self._links
    
    @property
    def message(self) -> str:
        return self._message
        
    def __str__(self):
        return self.message

class CompliancyCheck:
    
    def __init__(self):
        self.compliant: Optional[bool] = None
        self.warnings: List[Message] = []
        self.errors: List[Message] = []
        self.notes: List[Message] = []
    
    def warn(self, msg: Message):
        self.warnings.append(msg)
        
    def error(self, msg: Message):
        self.errors.append(msg)
        
    def note(self, msg: Message):
        self.notes.append(msg)
        
    def is_compliant(self):
        return len(self.errors) == 0
    
    def print(self):
        # Print all errors in red, with links in gray
        if self.errors:
            print(f"{Fore.RED}{Style.BRIGHT}Errors:")
            for error in self.errors:
                print(f"{Fore.RED}- {error.text}")
                for link in error.links:
                    print(f"  {Fore.LIGHTBLACK_EX}{link}")
            print()  # Line break
        
        # Print all warnings in yellow, with links in gray
        if self.warnings:
            print(f"{Fore.YELLOW}{Style.BRIGHT}Warnings:")
            for warning in self.warnings:
                print(f"{Fore.YELLOW}- {warning.text}")
                for link in warning.links:
                    print(f"  {Fore.LIGHTBLACK_EX}{link}")
            print()  # Line break
        
        # Print all notices in white, with links in gray
        if self.notes:
            print(f"{Fore.WHITE}{Style.BRIGHT}Notes:")
            for note in self.notes:
                print(f"{Fore.WHITE}- {note.text}")
                for link in note.links:
                    print(f"  {Fore.LIGHTBLACK_EX}{link}")
            print()
            
# module singleton
compliancy_check = CompliancyCheck()
            
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
    # global compliancy_check

    model_path = os.path.join(base, model_name)

    # check if the model folder exists
    if not os.path.isdir(model_path):
        compliancy_check.error(Message(f"Model folder {model_path} does not exist", DocuRef.MODEL_FOLDER_STRUCTURE))
    
    # check if the model folder contains the following and no additional ressources
    # - /dockerfiles/Dockerfile
    # - /config/default.yml
    # - /utils
    # - /meta.json

    # check if the model folder contains a Dockerfile
    dockerfile_path = os.path.join(model_path, "dockerfiles", "Dockerfile")
    if not os.path.isfile(dockerfile_path):
        compliancy_check.error(Message(f"Model folder {model_path} does not contain a Dockerfile", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.DOCKERFILE]))
    
    # check if the model folder contains a default config
    config_path = os.path.join(model_path, "config", "default.yml")
    if not os.path.isfile(config_path):
        compliancy_check.error(Message(f"Model folder {model_path} does not contain a default workflow configuration", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.CONFIG]))
    
    # check if the model folder contains a utils folder
    # NOTE: utils is not mandatory, however, all MHub-IO modules must be inside the utils folder if they exist.
    #       we can check modified files for any *.py and demand they're inside the utils folder.
    utils_path = os.path.join(model_path, "utils")
    if not os.path.isdir(utils_path):
       compliancy_check.note(Message(f"Model folder {model_path} does not contain a utils folder. All custom MHub-IO Modules must be under that folder.", DocuRef.MHUBIO_MODULES))
    
    # check if the model folder contains a model.json
    model_json_path = os.path.join(model_path, "meta.json")
    if not os.path.isfile(model_json_path):
        compliancy_check.error(Message(f"Model folder {model_path} does not contain a meta.json", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.MODEL_META_JSON]))
    
    # check if the model folder contains a mhub.toml
    mhub_toml_path = os.path.join(model_path, "mhub.toml")
    if not os.path.isfile(mhub_toml_path):
        compliancy_check.error(Message(f"Model folder {model_path} does not contain a mhub.toml", [DocuRef.MODEL_FOLDER_STRUCTURE, DocuRef.MODEL_TEST_PROCEDURE]))

def validateModelMetaJson(model_meta_json_file: str):

    # check that file exists or print a warning that the check has been skipped
    if not os.path.isfile(model_meta_json_file):
        compliancy_check.warn(Message(f"Model meta json validation skipped because file does not exist."))
        return

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
        compliancy_check.error(Message(f"Model meta json is not compliant with the schema: {e.message}", DocuRef.MODEL_META_JSON))

def validateModelMetaJson_modelName(model_meta_json_file: str, model_name: str):

    # check that file exists or print a warning that the check has been skipped
    if not os.path.isfile(model_meta_json_file):
        compliancy_check.warn(Message(f"Check for correct model name in meta.json skipped because file does not exist."))
        return

    # load model meta json
    with open(model_meta_json_file, "r") as f:
        model_meta_json = json.load(f)

    # check that the model meta json contains a name field
    if not "name" in model_meta_json:
        compliancy_check.warn(Message(f"Check for correct model name in meta.json skipped because name field is missing."))
        return

    # check that the model name is correct
    if model_meta_json["name"] != model_name:
        compliancy_check.error(Message(f"Model name in meta.json does not match model name in folder structure: {model_meta_json['name']} != {model_name}", DocuRef.MODEL_META_JSON))


def validateMHubToml(base: str, model_name: str):
        
    # get model toml path
    model_toml_file = os.path.join(base, model_name, "mhub.toml")
    
    # check that file exists or print a warning that the check has been skipped
    if not os.path.isfile(model_toml_file):
        compliancy_check.warn(Message(f"Model toml validation skipped because file does not exist."))
        return
    
    # load model toml
    with open(model_toml_file, "r") as f:
        model_toml = toml.load(f)

    # load schema
    with open(os.path.join('.github', 'schemas', 'mhubtoml.schema.json'), "r") as f:
        schema = json.load(f)
        
    # validate schema
    try:
        jsonschema.validate(instance=model_toml, schema=schema)
    except jsonschema.ValidationError as e:
        compliancy_check.error(Message(f"Model toml is not compliant with the schema: {e.message}", DocuRef.MODEL_TEST_PROCEDURE))
        
     
def validateModelTestData(base: str, model_name: str):
    
    # get model toml path
    model_toml_file = os.path.join(base, model_name, "mhub.toml")
    
    # check that file exists or print a warning that the check has been skipped
    if not os.path.isfile(model_toml_file):
        compliancy_check.warn(Message(f"Check for test data validation skipped because mhub.toml does not exist."))
        return
    
    # load model toml
    with open(model_toml_file, "r") as f:
        model_toml = toml.load(f)
        
    # check that the model toml contains a deployment test section
    if not "model" in model_toml or not "deployment" in model_toml["model"] or not "test" in model_toml["model"]["deployment"]:
        compliancy_check.error(Message(f"Model toml does not contain a deployment test section", DocuRef.MODEL_META_JSON))
        compliancy_check.warn(Message(f"Skipping additional test file checks because no test data url provided"))
        return
        
    # find test data url
    test_url = model_toml["model"]["deployment"]["test"]   
    
    # url has to be a valid url and point to zenodo
    if not test_url.startswith("https://zenodo.org/"):
        raise MHubComplianceError(f"Test data url is not a Zenodo url: {test_url}", DocuRef.MODEL_META_JSON)
    
    # get a list of all workflow names
    workflows = get_model_configuration_files(base, model_name)

    # Send a GET request to download the file
    response = requests.get(test_url, stream=True)

    # print status
    print(f"Downloading test data from {test_url}")
    print(f"Status code: {response.status_code}")
        
    # scan the zip file and check that it contains a folder for every config file and that each folder contains a sample and reference folder that are not empty
    with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
        for workflow in workflows:
            
            # check if the workflow folder exists
            if not f"{workflow}/" in zip_ref.namelist():
                compliancy_check.error(Message(f"Test data zip file does not contain a folder for workflow '{workflow}'", DocuRef.MODEL_META_JSON))
                compliancy_check.warn(Message(f"Skipping additional checks because no test files provided for workflow '{workflow}'"))
                continue
            
            # check if the workflow folder contains a sample and reference folder
            if not f"{workflow}/sample/" in zip_ref.namelist():
                compliancy_check.error(Message(f"Test data zip file does not contain a sample folder for workflow '{workflow}'", DocuRef.MODEL_META_JSON))
                compliancy_check.warn(Message(f"Skipping additional checks because no sample files provided for workflow '{workflow}'"))
                
            # if sample folder available, check if its empty
            elif len([p for p in zip_ref.namelist() if p.startswith(f"{workflow}/sample/")]) == 0:
                compliancy_check.error(Message(f"Test data zip file contains an empty sample folder for workflow '{workflow}'", DocuRef.MODEL_META_JSON))
                
            # check if the workflow folder contains a reference folder
            if not f"{workflow}/reference/" in zip_ref.namelist():
                compliancy_check.error(Message(f"Test data zip file does not contain a reference folder for workflow '{workflow}'", DocuRef.MODEL_META_JSON))
                compliancy_check.warn(Message(f"Skipping additional checks because no reference files provided for workflow '{workflow}'"))

            # if reference folder available, check if its empty                
            elif len([p for p in zip_ref.namelist() if p.startswith(f"{workflow}/reference/")]) == 0:
                compliancy_check.error(Message(f"Test data zip file contains an empty reference folder for workflow '{workflow}'", DocuRef.MODEL_META_JSON))
                

def validateDockerfile(base: str, model_name: str):
    
    # get dockerfile path
    model_dockerfile = os.path.join(base, model_name, "dockerfiles", "Dockerfile")

    # check that file exists or print a warning that the check has been skipped
    if not os.path.isfile(model_dockerfile):
        compliancy_check.warn(Message(f"Dockerfile validation skipped because file does not exist."))
        return

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
        compliancy_check.error(Message(f"Dockerfile does not contain the correct FROM command: {lines[0]}", DocuRef.DOCKERFILE))

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
            compliancy_check.error(Message(f"WORKDIR must not be set to any other than `/app` as defined in our base image. {line}", DocuRef.DOCKERFILE))

        if line.startswith("ADD") or line.startswith("COPY"):
            compliancy_check.error(Message(f"Dockerfile contains ADD or COPY command: {line}", DocuRef.DOCKERFILE))

        if line.startswith("FROM") and i > 0:
            compliancy_check.error(Message(f"Dockerfile contains FROM command not at the beginning of the file: {line}", DocuRef.DOCKERFILE))

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
                compliancy_check.error(Message(f"Collection import does not contain enough arguments: {line}", DocuRef.DOCKERFILE))
            
            # append to collections list
            dockerfile_mhubio_collections.append({
                "name": parts[2],
                "repo": parts[3] if len(parts) > 3 else None,
                "branch": parts[4] if len(parts) > 4 else None
            })

    # check if the dockerfile contains the required ARG MHUB_MODELS_REPO and model import
    if not dockerfile_defines_arg_mhub_models_repo:
        compliancy_check.error(Message(f"Dockerfile does not define 'ARG MHUB_MODELS_REPO'", DocuRef.DOCKERFILE))
    
    if not dockerfile_contains_mhubio_import:
        compliancy_check.error(Message(f"Dockerfile does not contain the required mhubio import command: 'RUN buildutils/import_mhub_model.sh {model_name} ${{MHUB_MODELS_REPO}}'", DocuRef.DOCKERFILE))
    
    # in case the model contains any mhubio-collection imports, check that they all are imported by theri name since custom imports are not allowed
    #  and also check that the model name is registered under base/collections.json
    if len(dockerfile_mhubio_collections) > 0:

        # load all registered officially suported collections
        with open(os.path.join('base', 'collections.json'), "r") as f:
            mhubio_collections = json.load(f)
            
        # check that all collections are registered
        for collection in dockerfile_mhubio_collections:
            if not collection["name"] in mhubio_collections:
                compliancy_check.error(Message(f"Collection '{collection['name']}' is unknown. We only allow official collections.", DocuRef.DOCKERFILE))

            # check that the collection is imported by its name
            if collection["repo"] is not None or collection["branch"] is not None:
                compliancy_check.error(Message(f"Collection '{collection['name']}' is not imported by its name.", DocuRef.DOCKERFILE))

    # check that the entrypoint of the dockerfile matches
    #  ENTRYPOINT ["mhub.run"]  |  ENTRYPOINT ["python", "-m", "mhubio.run"] (deprecated, no longer allowed)
    if not lines[-2].strip() == 'ENTRYPOINT ["mhub.run"]':
        compliancy_check.error(Message(f"Dockerfile does not contain the correct entrypoint: {lines[-2]}", DocuRef.DOCKERFILE))
    
    #  CMD ["--workflow", "default"] | CMD ["--config", "/app/models/$model_name/config/default.yml"]
    if not lines[-1].strip() in ['CMD ["--workflow", "default"]', f'CMD ["--config", "/app/models/{model_name}/config/default.yml"]']:
        compliancy_check.error(Message(f"Dockerfile does not contain the correct CMD command: {lines[-1]}", DocuRef.DOCKERFILE))
    

def get_model_configuration_files(base: str, model_name: str) -> List[str]:

    # get config path
    model_config_dir = os.path.join(base, model_name, "config")

    # get workflow files
    model_workflows = [cf[:-4] for cf in os.listdir(model_config_dir) if cf.endswith(".yml")]

    # return list of configuration files
    return model_workflows