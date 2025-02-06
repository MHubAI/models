"""
---------------------------------------------------------
Author: Leonard Nürnberg, Suraj Pai
Email:  lnuernberg@bwh.harvard.edu, bspai@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""

import json, jsonschema
from mhubio.core import Instance, InstanceData, InstanceDataCollection, IO, Module
import SimpleITK as sitk
import numpy as np

class CentroidExtractor(Module):

    @IO.Instance()
    @IO.Inputs('in_masks', 'nrrd|nifti:mod=seg', the='Tumor segmentation masks for the input NRRD files')
    @IO.Outputs('centroid_jsons', '[filename].json', "json:type=fmcibcoordinates", data='in_masks', the='JSON file containing 3D coordinates of the centroid of the input mask.')
    def task(self, instance: Instance, in_masks: InstanceDataCollection, centroid_jsons: InstanceDataCollection) -> None:
        for i, in_mask in enumerate(in_masks):
            seg_rois = in_mask.type.meta['roi'].split(',')
            mask = sitk.ReadImage(in_mask.abspath)
            mask_array = sitk.GetArrayFromImage(mask)
            unique_values = np.unique(mask_array)
            print(f"Unique values: {unique_values}")
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            seg_roi_coordinates = []

            for channel_id, seg_roi in enumerate(seg_rois):
                # Check if the label exists in the mask
                label = channel_id + 1
                if label not in unique_values:
                    print(f"Warning: Label {label} (ROI: {seg_roi}) not found in the mask. Skipping.")
                    continue

                # Calculate centroid if the label exists
                label_shape_filter.Execute(mask)
                try:
                    centroid = label_shape_filter.GetCentroid(label)
                    # Extract x, y, and z coordinates from the centroid
                    x, y, z = centroid

                    # Set up the coordinate dictionary
                    coordinate_dict = {
                        "Mhub ROI": seg_roi,
                        "coordX": x,
                        "coordY": y,
                        "coordZ": z,
                    }

                    seg_roi_coordinates.append(coordinate_dict)
                except Exception as e:
                    print(f"Error processing label {label} (ROI: {seg_roi}): {e}")

            centroid_json = centroid_jsons.get(i)
            # Write the coordinate dictionary to a json file
            with open(centroid_json.abspath, "w") as f:
                json.dump(seg_roi_coordinates, f)