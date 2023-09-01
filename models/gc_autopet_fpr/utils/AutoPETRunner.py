"""
-----------------------------------------------------------------------
Mhub / DIAG - Run Module for AutoPET false positive reduction algorithm
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""

from typing import List
from mhubio.core import Instance, DataTypeQuery, InstanceData, IO, Module

import os
import numpy as np
import SimpleITK as sitk

import torch

from process import Hybrid_cnn as Algorithm


# TODO should be moved to mhubio/core/templates.py
PT      = Meta(mod="PT")        # Positron emission tomography (PET)


class AutoPETRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'mha:mod=ct', the='input FDG CT scan')
    @IO.Input('in_data_pet', 'mha:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data', 'tumor_segmenation.mha', 'mha:mod=seg:model=AutoPET', bundle='model', the='predicted tumor segmentation within the input FDG PET/CT scan')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data: InstanceData) -> None:
        # TODO link in_data.abspath
        # TODO link out_data.abspath
        algorithm = Algorithm()
        algorithm.input_path = Path(in_data_ct.abspath).parent
        algorithm.output_path = Path(out_data.abspath).parent
        algorithm.process()
