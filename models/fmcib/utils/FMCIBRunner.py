"""
---------------------------------------------------------
Author: Suraj Pia
Email:  bspai@bwh.harvard.edu
---------------------------------------------------------
"""

import json
import torch
from fmcib.models import fmcib_model 
import SimpleITK as sitk
from mhubio.core import Instance, InstanceData, IO, Module
from fmcib.preprocessing import preprocess


class FMCIBRunner(Module):
    @IO.Instance()
    @IO.Input('in_data', 'nrrd:mod=ct', the='Input NRRD file')
    @IO.Input('in_mask', 'nrrd|json', the='Tumor mask for the input NRRD file')
    @IO.Output('feature_json', 'features.json', "json", bundle='model', the='output JSON file')
    def task(self, instance: Instance, in_data: InstanceData, in_mask: InstanceData, feature_json: InstanceData) -> None:
        mask_path = in_mask.abspath
        mask = sitk.ReadImage(mask_path)

        # Get the CoM of the mask
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        try:
            centroid = label_shape_filter.GetCentroid(255)
        except:
            centroid = label_shape_filter.GetCentroid(1)

        x, y, z = centroid

        input_dict = {
            "image_path": in_data.abspath,
            "coordX": x,
            "coordY": y,
            "coordZ": z,
        }

        image = preprocess(input_dict)
        image = image.unsqueeze(0)
        model = fmcib_model()

        model.eval()
        with torch.no_grad():
            features = model(image)

        feature_dict = {f"feature_{idx}": feature for idx, feature in enumerate(features.flatten().tolist())}

        with open(feature_json.abspath, "w") as f:
            json.dump(feature_dict, f)
