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

from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, ClassOutput, Meta

import SimpleITK

import tensorflow as tf

# Import the Luna22 baseline algorithm class from the luna22-ismi-algorithm repository
from process import Nodule_classifier as NoduleClassifier


@ValueOutput.Name('malignancy_risk')
@ValueOutput.Meta(Meta(key="value"))
@ValueOutput.Label('MalignancyRisk')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of the nodule malignancy.')
class MalignancyRiskOutput(ValueOutput):
    pass


@ClassOutput.Name('texture')
@ClassOutput.Label('NoduleType')
@ClassOutput.Description('Prediction of the nodule type.')
@ClassOutput.Class(0, 'Non-solid', 'Non-solid.')
@ClassOutput.Class(1, 'Part-solid', 'Part-solid.')
@ClassOutput.Class(2, 'Solid', 'Solid.')
class NoduleTypeOutput(ClassOutput):
    pass


class Luna22IsmiBaselineRunner(Module):

    _cached_classifier: Optional[NoduleClassifier] = None

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=ct', the='input volume centered around a nodule from a lung CT image')
    @IO.OutputData('malignancy_risk', MalignancyRiskOutput, data='in_data', the='Luna22 Ismi nodule malignancy probability')
    @IO.OutputData('texture', NoduleTypeOutput, data='in_data', the='Luna22 Ismi nodule type classification')
    def task(self, instance: Instance, in_data: InstanceData, malignancy_risk: MalignancyRiskOutput, texture: NoduleTypeOutput) -> None:
        assert len(tf.config.list_physical_devices("GPU")) >= 1, \
            "Error: Luna22IsmiBaselineRunner must be run on a GPU! Because: " \
            "The Conv2D op currently only supports the NHWC tensor format on" \
            " the CPU. The op was given the format: NCHW [Op:Conv2D]"

        # Read input image
        input_image = SimpleITK.ReadImage(in_data.abspath)

        # Create classifier and load model weights if not already loaded
        if self._cached_classifier is None:
            self.v("Luna22IsmiBaselineRunner - Loading and caching model weights")
            self._cached_classifier = NoduleClassifier()
        classifier = self._cached_classifier

        # Run the classifier on the input image
        predictions = classifier.predict(input_image=input_image)

        # Output the predicted values
        malignancy_risk.value = predictions["malignancy_risk"]
        texture.value = predictions["texture"]
