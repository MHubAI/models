"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Spider baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module

import SimpleITK
import json

from process import SpiderAlgorithm


class SpiderBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=mri', the='input sagittal spine MRI')
    @IO.Output('out_data', 'spider_baseline_vertebrae_segmentation.mha', 'json:model=SpiderBaselineS', 'in_data', the='Spider baseline vertebrae segmentation')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        input_image = SimpleITK.ReadImage(in_data.abspath)
        predictions = SpiderAlgorithm().predict(input_image=input_image)
        predictions = {k.name:v for k,v in predictions.items()}
        with open(out_data.abspath, "w") as f:
            json.dump(predictions, f, indent=4)
