"""
---------------------------------------------------------
Author: Leonard Nürnberg
Email:  lnuernberg@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""

import json, jsonschema
from mhubio.core import Instance, InstanceData, InstanceDataCollection, IO, Module
import SimpleITK as sitk

class CentroidExtractor(Module):
  
    @IO.Instance()
    @IO.Inputs('in_masks', 'nrrd|nifti:mod=seg', the='Tumor segmentation masks for the input NRRD files')
    @IO.Outputs('centroid_jsons', '[filename].json', "json:type=fmcibcoordinates", data='in_masks', the='JSON file containing 3D coordinates of the centroid of the input mask.')
    def task(self, instance: Instance, in_masks: InstanceDataCollection, centroid_jsons: InstanceDataCollection) -> None:
        for i, in_mask in enumerate(in_masks):
            seg_rois = in_mask.type.meta['roi'].split(',')
            mask = sitk.ReadImage(in_mask.abspath)
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            seg_roi_coordinates = []
            
            for channel_id, seg_roi in enumerate(seg_rois):
                # get the center of massk from the mask via ITK
                label_shape_filter.Execute(mask)
                centroid = label_shape_filter.GetCentroid(channel_id + 1)

                # extract x, y, and z coordinates from the centroid
                x, y, z = centroid
                
                # set up the coordinate dictionary
                coordinate_dict = {
                    "Mhub ROI": seg_roi,
                    "coordX": x,
                    "coordY": y,
                    "coordZ": z,
                }

                seg_roi_coordinates.append(coordinate_dict)


            centroid_json = centroid_jsons.get(i)
            # write the coordinate dictionary to a json file
            with open(centroid_json.abspath, "w") as f:
                json.dump(seg_roi_coordinates, f)
