"""
-------------------------------------------------
MHub - Skull Stripping Module
-------------------------------------------------

-------------------------------------------------
Author: Jithendra
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from enum import Enum
from typing import List, Dict, Any
from pathlib import Path
from mhubio.core import Module, Instance, InstanceDataCollection, InstanceData, DataType, FileType
from mhubio.core.IO import IO

import os, subprocess

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be skull stripped")
class SkullStripRunner(Module):
    """
    Skull Strip images
    """

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:task=skull_stripped', data='in_datas', bundle='skull_stripping', auto_increment=True, the="converted data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, out_datas: InstanceDataCollection, **kwargs) -> None:

        # some sanity checks
        assert isinstance(in_datas, InstanceDataCollection)
        assert isinstance(out_datas, InstanceDataCollection)
        assert len(in_datas) == len(out_datas)

        # filtered collection must not be empty
        if len(in_datas) == 0:
            self.v(f"CONVERT ERROR: no data found in instance {str(instance)}.")
            return None

        # conversion step
        for i, in_data in enumerate(in_datas):
            out_data = out_datas.get(i)
            synth_cmd = [
                "mri_synthstrip",
                "-i",
                str(in_data.abspath),
                "-o",
                str(out_data.abspath),
            ]
            self.subprocess(synth_cmd, text=True)
