"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Node21 baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module, Meta

import SimpleITK
import json
from pathlib import Path

from process import Noduledetection

# TODO should move to mhubio/core/templates.py
CR      = Meta(mode="CR")       # CR	Computed Radiography

class Node21BaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=cr', the='input chest X-Ray')
    @IO.Output('out_data', 'nodules.json', 'json:model=Node21Baseline', 'in_data', the='Stoic baseline bounding box nodule predictions')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        input_image = SimpleITK.ReadImage(in_data.abspath)
        tmp_path = Path("/app/tmp")
        predictions = Noduledetection(input_dir=tmp_path, output_dir=tmp_path).predict(input_image=input_image)
        with open(out_data.abspath, "w") as f:
            json.dump(predictions, f, indent=4)
