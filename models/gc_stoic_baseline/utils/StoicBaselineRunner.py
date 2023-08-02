"""
------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Stoic baseline Algorithm
------------------------------------------------------------

------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module

import SimpleITK
import json

from process import StoicAlgorithm


class StoicBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=ct', the='input lung CT')
    @IO.Output('out_data', 'gc_stoic_baseline_covid_scores.json', 'json:model=StoicBaselineCovidScore', 'in_data', the='Stoic baseline covid 19 scores')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        input_image = SimpleITK.ReadImage(in_data.abspath)
        predictions = StoicAlgorithm().predict(input_image=input_image)
        predictions = {k.name:v for k,v in predictions.items()}
        with open(out_data.abspath, "w") as f:
            json.dump(predictions, f, indent=4)
