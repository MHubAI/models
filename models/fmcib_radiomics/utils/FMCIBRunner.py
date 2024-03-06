"""
---------------------------------------------------------
Author: Suraj Pia
Email:  bspai@bwh.harvard.edu
---------------------------------------------------------
"""

import json, jsonschema, os
from fmcib.models import fmcib_model 
import SimpleITK as sitk
from mhubio.core import Instance, InstanceData, IO, Module

COORDS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "coords.schema.json")

def fmcib(input_dict: dict, json_output_file_path: str):
    """Run the FCMIB pipeline.

    Args:
        input_dict (dict): The input dictionary containing the image path and the seed point coordinates.
        json_output_file_path (str): The path were the features are exported to as a json file.
    """
    # model dependency imports
    import torch
    from fmcib.preprocessing import preprocess
    
    # initialize model
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
    @IO.Input('in_data', 'nrrd:mod=ct', the='Input NRRD file')
    @IO.Input('centroids_json', 'json:type=fmcibcoordinates', the='The centroids in the input image coordinate space')
    @IO.Output('feature_json', 'features.json', "json:type=fmcibfeatures", bundle='model', the='Features extracted from the input image at the specified seed point.')
    def task(self, instance: Instance, in_data: InstanceData, centroids_json: InstanceData, feature_json: InstanceData) -> None:
        
        # read centroids from json file
        centroids = json.load(centroids_json.abspath)

        # verify input data schema
        with open("models/fmcib_radiomics/utils/input_schema.json") as f:
            schema = json.load(f)
        jsonschema.validate(centroids, schema)

        # define input dictionary
        input_dict = {
            "image_path": in_data.abspath,
            **centroids
        }

        # run model
        fmcib(input_dict, feature_json.abspath)