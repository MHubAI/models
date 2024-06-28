"""
---------------------------------------------------------
Author: Suraj Pai, Leonard Nürnberg 
Email:  bspai@bwh.harvard.edu, lnuernberg@bwh.harvard.edu
Date:   06.03.2024
---------------------------------------------------------
"""
import os
from pathlib import Path
from mhubio.core import Instance, InstanceData, IO, Module
from bpreg.inference.inference_model import InferenceModel
import torch


class BpRegRunner(Module):
    """
    BpRegRunner is a module for running body part regression on CT images.
    It takes a NIfTI CT image as input and outputs a JSON file with extracted features.
    """
    
    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=ct', the='Input nifti ct image file')
    @IO.Output('utility_json', 'utility.json', "json:type=utility", bundle='model', the='Features extracted from the input image')
    def task(self, instance: Instance, in_data: InstanceData, utility_json: InstanceData) -> None:
        """
        Perform body part regression on the input NIfTI CT image and save the extracted features to a JSON file.

        Args:
            instance (Instance): The instance of the module.
            in_data (InstanceData): The input NIfTI CT image file.
            utility_json (InstanceData): The output JSON file to save the extracted features.
        """
        gpu_available = torch.cuda.is_available()
        model = InferenceModel("/app/public_inference_model/public_bpr_model/", gpu=gpu_available)
        input_path = in_data.abspath
        output_path = utility_json.abspath
        model.nifti2json(input_path, output_path, stringify_json=False)