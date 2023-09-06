"""
------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Stoic baseline Algorithm
------------------------------------------------------------

------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput

import SimpleITK
import json

from process import StoicAlgorithm


@ValueOutput.Name('probability-covid-19')
@ValueOutput.Meta(Meta(key="value"))
@ValueOutput.Label('Covid19Probability')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of presence of Covid19.')
class Covid19ProbabilityOutput(ValueOutput):
    pass


@ValueOutput.Name('probability-severe-covid-19')
@ValueOutput.Meta(Meta(key="value"))
@ValueOutput.Label('SevereCovid19Probability')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of presence of severe Covid19.')
class SevereCovid19ProbabilityOutput(ValueOutput):
    pass


class StoicBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=ct', the='input lung CT')
    @IO.OutputData('probability_covid_19', Covid19ProbabilityOutput, data='in_data', the='Stoic baseline Covid19 probability')
    @IO.OutputData('probability_severe_covid_19', SevereCovid19ProbabilityOutput, data='in_data', the='Stoic baseline severe Covid19 probability')
    def task(self, instance: Instance, in_data: InstanceData, probability_covid_19: Covid19ProbabilityOutput, probability_severe_covid_19: SevereCovid19ProbabilityOutput) -> None:
        input_image = SimpleITK.ReadImage(in_data.abspath)
        predictions = StoicAlgorithm().predict(input_image=input_image)
        predictions = {k.name:v for k,v in predictions.items()}
        # configure output data
        probability_covid_19.value = predictions["probability-covid-19"]
        probability_severe_covid_19.value = predictions["probability-severe-covid-19"]
