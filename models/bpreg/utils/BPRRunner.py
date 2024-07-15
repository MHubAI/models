"""
---------------------------------------------------------
Author: Suraj Pai, Leonard Nürnberg 
Email:  bspai@bwh.harvard.edu, lnuernberg@bwh.harvard.edu
Date:   15.07.2024
---------------------------------------------------------
"""
import os
from pathlib import Path
from mhubio.core import Instance, InstanceData, IO, Module
from bpreg.inference.inference_model import InferenceModel
import torch


class BPRRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=ct', the='The input NIfTI CT image file.')
    @IO.Output('bpreg_json', 'bpreg.json', "json:type=bpreg", bundle='model', the=' The output JSON file to save the extracted features.')
    def task(self, instance: Instance, in_data: InstanceData, utility_json: InstanceData) -> None:
        gpu_available = torch.cuda.is_available()
        model = InferenceModel("/app/public_inference_model/public_bpr_model/", gpu=gpu_available)
        input_path = in_data.abspath
        output_path = utility_json.abspath
        model.nifti2json(input_path, output_path, stringify_json=False)