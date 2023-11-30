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


@ValueOutput.Name('prostate_cancer_likelihood')
@ValueOutput.Label('ProstateCancerLikelihood')
@ValueOutput.Type(float)
@ValueOutput.Description('Likelihood of case-level prostate cancer.')
class ProstateCancerLikelihood(ValueOutput):
    pass


class PicaiBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_t2', 'mha:mod=mr:type=t2w', the='input T2 weighted prostate MR image')
    @IO.Input('in_data_adc', 'mha:mod=mr:type=adc', the='input ADC prostate MR image')
    @IO.Input('in_data_hbv', 'mha:mod=mr:type=hbv', the='input HBV prostate MR image')
    @IO.Output('cancer_likelihood_json', 'cspca-case-level-likelihood.json', "json", bundle='model', the='output JSON file with PICAI baseline prostate cancer likelihood')
    @IO.Output('cancer_detection_heatmap', 'cspca_detection_map.mha', "mha:mod=hm", bundle='model', the='output heatmap indicating prostate cancer likelihood')
    @IO.OutputData('cancer_likelihood', ProstateCancerLikelihood, the='PICAI baseline prostate cancer likelihood')
    def task(self, instance: Instance, in_data_t2: InstanceData, in_data_adc: InstanceData, in_data_hbv: InstanceData, cancer_likelihood_json: InstanceData, cancer_detection_heatmap: InstanceData, cancer_likelihood: ProstateCancerLikelihood) -> None:
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
        classifier.case_confidence_path = Path(cancer_likelihood_json.abspath)

        # Run the classifier on the input images
        classifier.process()

        # Extract cancer likelihood value from cancer_likelihood_file
        if not Path(cancer_likelihood_json.abspath).is_file():
            raise FileNotFoundError(f"Output file {cancer_likelihood_json.abspath} could not be found!")

        with open(cancer_likelihood_json.abspath, "r") as f:
            cancer_lh = float(json.load(f))

        if not (isinstance(cancer_lh, (float, int)) and (0.0 <= cancer_lh <= 1.0)):
            raise ValueError(f"Cancer likelihood value should be between 0 and 1, found: {cancer_lh}")

        # Output the predicted values
        cancer_likelihood.value = cancer_lh
