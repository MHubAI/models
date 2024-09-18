import utils
import os, sys, json

# introduction
print()
print("------------------------------------------------")
print("MHub Compliance Checks started.")
print("We will check for a correct folder setup, Dockerfile and meta.json compliance.")
print()

# print event path variable
PR = os.environ['PR_NUMBER']
print("Pull request:    ", PR)

# get the first argument to this script which should be the list of modified files from an earlyer step
modified_files = json.loads(os.environ['MODIFIED_FILES']) 
print("Modified files:  ", "\n                  ".join(modified_files))

# modified models list
modified_models = list(set(fp.split("/")[1] for fp in modified_files))
print("Modified models: ", ", ".join(modified_models))

# we allow modifications only to a single file for now
# TODO: iterate model list (we can outsource model checks and then call a check_model script with the model name as argument)
if len(modified_models) != 1:
    print("CHECK FAILED: ", "Exactly one model must be modified in a pull request.")
    sys.exit(1)

# model name
model_name = modified_models[0]

# run compliance checks
try:
    # check folder structure
    utils.validateModelFolder(base='models', model_name=model_name)

    # check meta.json (schema)
    utils.validateModelMetaJson(model_meta_json_file=os.path.join('models', model_name, 'meta.json'))

    # check additional requirements for meta.json
    utils.validateModelMetaJson_modelName(model_meta_json_file=os.path.join('models', model_name, 'meta.json'), model_name=model_name)

    # validate dockerfile
    utils.validateDockerfile(base='models', model_name=model_name)
    
    # validate the mhub.toml file
    utils.validateMHubToml(base='models', model_name=model_name)
    
    # validate mhub model test data
    utils.validateModelTestData(base='models', model_name=model_name)

except Exception as e:
    print()
    print("--------------- CHECK DISRUPTED --------------")
    print("An unexpected error occured during compliance checks.")
    print(str(e))
    print()
    sys.exit(1)
    
# check compliance check output
if utils.compliancy_check.is_compliant():
    print()
    print("---------------- CHECK PASSED ----------------")
    print("All compliance checks passed.")
    print("Note: compliance checks are a beta feature. Passing all automated compliance checks does not guarantee that your model is compliant with the MHub standard. We will now perform a manual review of your model. Testing your model on a public dataset is obligatory.")
    print()
    
else:
    print()
    print("---------------- CHECK FAILED ----------------")
    utils.compliancy_check.print()
    sys.exit(1)
