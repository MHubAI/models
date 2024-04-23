import sys, os, yaml, json, jsonschema

YAML_TEST_DEFINITION_SCHEMA_FILE = ".github/schemas/testmodel.schema.json"

def extract_yaml_test_definition(comment: str):
  
  # find a code block starting with ```yaml and ending with ```
  start = comment.find("```yaml")
  end = comment.find("```", start + 1)
  if start == -1 or end == -1:
    raise Exception("No YAML code block found in comment")
  
  # extract the code block
  yaml_code = comment[start:end]
  
  # remove the code block markers
  yaml_code = yaml_code.replace("```yaml", "").strip()
  
  return yaml_code

def validate_yaml_test_definition(yaml_str: str):

  # load yaml into dict
  test_definition = yaml.safe_load(yaml_str)
  
  # load schema
  with open(YAML_TEST_DEFINITION_SCHEMA_FILE, "r") as f:
    schema = json.load(f)
    
  # validate
  jsonschema.validate(test_definition, schema)
  
  
def set_action_output(output_name, value) :
    """Sets the GitHub Action output.

    Keyword arguments:
    output_name - The name of the output
    value - The value of the output
    """
    if "GITHUB_OUTPUT" in os.environ :
        with open(os.environ["GITHUB_OUTPUT"], "a") as f :
            print("{0}={1}".format(output_name, value), file=f)

    
if __name__ == "__main__":
  
  try:
    # get comment body from first argument
    comment = sys.argv[1]
    
    # print comment
    print(f"Comment ----------------------")
    print(comment)
    print()
    
    # extract yaml test definition
    yaml_str = extract_yaml_test_definition(comment)
    
    # validate yaml test definition
    validate_yaml_test_definition(yaml_str)
    
    # print yaml
    print(f"Test Definition --------------")
    print(yaml_str)
    print()
    
    # print success message
    print("YAML test definition is valid")

    # set environment variable for following steps
    set_action_output("test_report", "passed")
    
  except Exception as e:
    # set environment variable for following steps
    set_action_output("test_report", "failed")
    
    # print error message
    print("YAML test definition is invalid")
    print(e)
    
