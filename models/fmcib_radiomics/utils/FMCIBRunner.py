"""
---------------------------------------------------------
Author: Suraj Pai, Leonard Nürnberg 
Email:  bspai@bwh.harvard.edu, lnuernberg@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""
import json, jsonschema, os
import pandas as pd
from mhubio.core import Instance, InstanceData, InstanceDataCollection, IO, Module
from mhubio.modules.organizer.DataOrganizer import DataOrganizer

COORDS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "coords.schema.json")
ROI_COORDS_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "roi.coords.schema.json")
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
        yield json_data    
    
    elif is_valid(json_data, ROI_COORDS_SCHEMA_PATH):
        for item in json_data:
            yield item


    elif is_valid(json_data, SLICERMARKUP_SCHEMA_PATH):
        markups = json_data["markups"]

        assert len(markups) == 1, "Currently, only one point per file is supported."
        markup = markups[0]
        assert markup["coordinateSystem"] == "LPS"
        for controlPoint in markup["controlPoints"]:
            position = controlPoint["position"]
            yield {
                "coordX": position[0],
                "coordY": position[1],
                "coordZ": position[2]
            }
            
    else:
        raise ValueError("The input json file does not adhere to the expected schema.")
    
def fmcib(input_dict: dict):
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
    return feature_dict

class FMCIBRunner(Module):
    
    @IO.Instance()
    @IO.Input('in_data', 'nrrd|nifti:mod=ct', the='Input nrrd or nifti ct image file')
    @IO.Inputs('centroid_jsons', "json:type=fmcibcoordinates", the='JSON file containing 3D coordinates of the centroid of the input mask.')
    @IO.Output('feature_csv', '[i:sid]/features.csv', 'csv:features=fmcib', data='in_data',  bundle='model', the='Features extracted from the input image at the specified seed point.')
    def task(self, instance: Instance, in_data: InstanceData, centroid_jsons: InstanceDataCollection, feature_csv: InstanceData) -> None:
        for centroid_json in centroid_jsons:
            # read centroids from json file
            roi_feature_list = []
            for roi_idx, coord_dict in enumerate(get_coordinates(centroid_json.abspath)):
                if "Mhub ROI" not in coord_dict:
                    coord_dict["Mhub ROI"] = roi_idx
                # define input dictionary
                input_dict = {
                    "image_path": in_data.abspath, 
                    "coordX": coord_dict["coordX"],
                    "coordY": coord_dict["coordY"],
                    "coordZ": coord_dict["coordZ"]
                }

                # run model
                feature_dict = fmcib(input_dict)
                feature_dict["Mhub ROI"] = coord_dict["Mhub ROI"]
                feature_dict["Mask"] = centroid_json.abspath.split("/")[-1]
                roi_feature_list.append(feature_dict)


            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(roi_feature_list)
            # Save the DataFrame to a CSV file

            df.to_csv(feature_csv.abspath, index=False)
