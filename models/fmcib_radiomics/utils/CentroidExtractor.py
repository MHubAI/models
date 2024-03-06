"""
---------------------------------------------------------
Author: Leonard Nürnberg
Email:  lnuernberg@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""

import json, jsonschema
from mhubio.core import Instance, InstanceData, IO, Module
import SimpleITK as sitk

class CentroidExtractor(Module):
  
    @IO.Instance()
    @IO.Input('in_mask', 'nrrd:mod=seg', the='Tumor segmentation mask for the input NRRD file.')
    @IO.Output('centroids_json', 'centroids.json', "json:type=fmcibcoordinates", the='JSON file containing 3D coordinates of the centroid of the input mask.')
    def task(self, instance: Instance, in_data: InstanceData, in_mask: InstanceData, centroids_json: InstanceData) -> None:
        
        # read the input mask 
        mask = sitk.ReadImage(in_mask.abspath)

        # get the center of massk from the mask via ITK
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        try:
            centroid = label_shape_filter.GetCentroid(255)
        except:
            centroid = label_shape_filter.GetCentroid(1)

        # extract x, y, and z coordinates from the centroid
        x, y, z = centroid

        # set up the coordinate dictionary
        coordinate_dict = {
            "coordX": x,
            "coordY": y,
            "coordZ": z,
        }

        # write the coordinate dictionary to a json file
        with open(centroids_json.abspath, "w") as f:
            json.dump(coordinate_dict, f)
