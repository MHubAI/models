"""
------------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Luna22 ISMI baseline Algorithm
------------------------------------------------------------------

------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import Optional

from mhubio.core import Instance, InstanceData, IO, Module

import SimpleITK

from process import Nodule_classifier as NoduleClassifier

import tensorflow as tf


class Luna22IsmiBaselineRunner(Module):

    _cached_classifier: Optional[NoduleClassifier] = None

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=ct', the='input volume centered around a nodule from a lung CT image')
    @IO.Output('out_data', 'nodule_predictions.json', 'json:model=Luna22IsmiBaseline', 'in_data', the='Luna22 Ismi nodule predictions (malignancy / nodule type)')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        assert len(tf.config.list_physical_devices("GPU")) >= 1, \
            "Error: Luna22IsmiBaselineRunner must be run on a GPU! Because: " \
            "The Conv2D op currently only supports the NHWC tensor format on" \
            " the CPU. The op was given the format: NCHW [Op:Conv2D]"

        input_image = SimpleITK.ReadImage(in_data.abspath)
        if self._cached_classifier is None:
            self.v("Luna22IsmiBaselineRunner - Loading and caching model weights")
            self._cached_classifier = NoduleClassifier()
        classifier = self._cached_classifier
        predictions = classifier.predict(input_image=input_image)
        with open(out_data.abspath, "w") as f:
            json.dump(predictions, f, indent=4)
