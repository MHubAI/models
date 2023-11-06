"""
---------------------------------------------------------
Mhub / DIAG - Run Module for the PICAI baseline Algorithm
---------------------------------------------------------

---------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
---------------------------------------------------------
"""

import json
from pathlib import Path

from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, ClassOutput, Meta

# Import the PICAI Classifier algorithm class from /opt/algorithm
from process import csPCaAlgorithm as PicaiClassifier


@ValueOutput.Name('prostate_cancer_probability')
@ValueOutput.Meta(Meta(key="value"))
@ValueOutput.Label('ProstateCancerProbability')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of case-level prostate cancer.')
class ProstateCancerProbability(ValueOutput):
    pass


class PicaiBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_t2', 'mha:mod=mr:type=t2w', the='input T2 weighted prostate MR image')
    @IO.Input('in_data_adc', 'mha:mod=mr:type=adc', the='input ADC prostate MR image')
    @IO.Input('in_data_hbv', 'mha:mod=mr:type=hbv', the='input HBV prostate MR image')
    @IO.Output('cancer_probability_json', 'cspca-case-level-likelihood.json', "json", bundle='model', the='output JSON file with PICAI baseline prostate cancer probability')
    @IO.Output('cancer_detection_heatmap', 'cspca_detection_map.mha', "mha:mod=hm", bundle='model', the='output heatmap indicating prostate cancer probability')
    @IO.OutputData('cancer_probability', ProstateCancerProbability, the='PICAI baseline prostate cancer probability')
    def task(self, instance: Instance, in_data_t2: InstanceData, in_data_adc: InstanceData, in_data_hbv: InstanceData, cancer_probability_json: InstanceData, cancer_detection_heatmap: InstanceData, cancer_probability: ProstateCancerProbability) -> None:
        # Initialize classifier object
        classifier = PicaiClassifier()

        # Specify input files (the order is important!)
        classifier.scan_paths = [
            Path(in_data_t2.abspath),
            Path(in_data_adc.abspath),
            Path(in_data_hbv.abspath),
        ]

        # Specify output files
        classifier.cspca_detection_map_path = Path(cancer_detection_heatmap.abspath)
        classifier.case_confidence_path = Path(cancer_probability_json.abspath)

        # Run the classifier on the input images
        classifier.process()

        # Extract cancer probability value from cancer_probability_file
        if not Path(cancer_probability_json.abspath).is_file():
            raise FileNotFoundError(f"Output file {cancer_probability_json.abspath} could not be found!")

        with open(cancer_probability_json.abspath, "r") as f:
            cancer_prob = float(json.load(f))

        if not (isinstance(cancer_prob, (float, int)) and (0.0 <= cancer_prob <= 1.0)):
            raise ValueError(f"Cancer probability value should be a probability value, found: {cancer_prob}")

        # Output the predicted values
        cancer_probability.value = cancer_prob
