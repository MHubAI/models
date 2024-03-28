"""
------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Stoic baseline Algorithm
------------------------------------------------------------

------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------------
"""
import os
from contextlib import contextmanager
from pathlib import Path

import SimpleITK

from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, Meta

# Import the StoicAlgorithm which was installed from the stoic2021-baseline repo
from process import StoicAlgorithm


# Retrieve STOIC source path from the environment variable
STOIC_SRC_PATH = Path(os.environ["STOIC_SRC_PATH"])


@ValueOutput.Name('probability-covid-19')
@ValueOutput.Label('Covid19Probability')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of presence of Covid19.')
class Covid19ProbabilityOutput(ValueOutput):
    pass


@ValueOutput.Name('probability-severe-covid-19')
@ValueOutput.Label('SevereCovid19Probability')
@ValueOutput.Type(float)
@ValueOutput.Description('Probability of presence of severe Covid19.')
class SevereCovid19ProbabilityOutput(ValueOutput):
    pass


@contextmanager
def set_directory(path: Path):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


class StoicBaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=ct', the='input lung CT')
    @IO.OutputData('probability_covid_19', Covid19ProbabilityOutput, data='in_data', the='Stoic baseline Covid19 probability')
    @IO.OutputData('probability_severe_covid_19', SevereCovid19ProbabilityOutput, data='in_data', the='Stoic baseline severe Covid19 probability')
    def task(self, instance: Instance, in_data: InstanceData, probability_covid_19: Covid19ProbabilityOutput, probability_severe_covid_19: SevereCovid19ProbabilityOutput) -> None:
        # Read input image
        input_image = SimpleITK.ReadImage(in_data.abspath)

        # Run the STOIC baseline algorithm on the input_image and retrieve the predictions
        # Set workdir to STOIC_SRC_PATH to allow algorithm to pick up model weights correctly
        with set_directory(STOIC_SRC_PATH):
            predictions = StoicAlgorithm().predict(input_image=input_image)
            predictions = {k.name:v for k,v in predictions.items()}

        # Configure the output data using the predictions
        probability_covid_19.value = predictions["probability-covid-19"]
        probability_severe_covid_19.value = predictions["probability-severe-covid-19"]
