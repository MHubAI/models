"""
---------------------------------------------------------
Author: Suraj Pai, Leonard Nürnberg 
Email:  bspai@bwh.harvard.edu, lnuernberg@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""
import json, jsonschema, os
from mhubio.core import Instance, InstanceData, IO, Module

COORDS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "coords.schema.json")
SLICERMARKUP_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "slicermarkup.schema.json")

def is_valid(json_data: dict, schema_file_path: str) -> bool:
    """Check if a json file is valid according to a given schema.

    Args:
        json_data (dict): The json data to be validated.
        schema_file_path (str): The path to the schema file.

    Returns:
        bool: True if the json file is valid according to the schema, False otherwise.
    """
    with open(schema_file_path) as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(json_data, schema)
        return True
    except:
        return False

def get_coordinates(json_file_path: str) -> dict:
    
    # read json file
    with open(json_file_path) as f:
        json_data = json.load(f)
        
    # check which schema the json file adheres to
    if is_valid(json_data, COORDS_SCHEMA_PATH):
        return json_data
    
    if is_valid(json_data, SLICERMARKUP_SCHEMA_PATH):
        markups = json_data["markups"]

        assert len(markups) == 1, "Currently, only one point per file is supported."
        markup = markups[0]
        
        assert markup["coordinateSystem"] == "LPS"
        
        controlPoints = markup["controlPoints"]
        assert len(controlPoints) == 1
        
        position = controlPoints[0]["position"]
        return {
            "coordX": position[0],
            "coordY": position[1],
            "coordZ": position[2]
        }
        
    #
    raise ValueError("The input json file does not adhere to the expected schema.")
    
def fmcib(input_dict: dict, json_output_file_path: str):
    """Run the FCMIB pipeline.

    Args:
        input_dict (dict): The input dictionary containing the image path and the seed point coordinates.
        json_output_file_path (str): The path were the features are exported to as a json file.
    """
    # model dependency imports
    import torch
    from fmcib.models import fmcib_model 
    from fmcib.preprocessing import preprocess
    
    # initialize the ResNet50 model with pretrained weights
    model = fmcib_model()
    
    # run model preroecessing
    image = preprocess(input_dict)
    image = image.unsqueeze(0)
    
    # run model inference
    model.eval()
    with torch.no_grad():
        features = model(image)

    # generate fearure dictionary
    feature_dict = {f"feature_{idx}": feature for idx, feature in enumerate(features.flatten().tolist())}

    # write feature dictionary to json file
    with open(json_output_file_path, "w") as f:
        json.dump(feature_dict, f)

class FMCIBRunner(Module):
    
    @IO.Instance()
    @IO.Input('in_data', 'nrrd|nifti:mod=ct', the='Input nrrd or nifti ct image file')
    @IO.Input('centroids_json', "json:type=fmcibcoordinates", the='JSON file containing 3D coordinates of the centroid of the input mask.')
    @IO.Output('feature_json', 'features.json', "json:type=fmcibfeatures", bundle='model', the='Features extracted from the input image at the specified seed point.')
    def task(self, instance: Instance, in_data: InstanceData, centroids_json: InstanceData, feature_json: InstanceData) -> None:
        
        # read centroids from json file
        coordinates = get_coordinates(centroids_json.abspath)

        # define input dictionary
        input_dict = {
            "image_path": in_data.abspath,
            **coordinates
        }


        # run model
        fmcib(input_dict, feature_json.abspath)