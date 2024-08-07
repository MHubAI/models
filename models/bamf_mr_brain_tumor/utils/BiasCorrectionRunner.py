"""
-------------------------------------------------
MHub - Bias Correction Module
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

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be bias corrected")
@IO.Config('bundle_name', str, 'bias_correction', the="bundle name converted data will be added to")
@IO.Config('converted_file_name', str, '[filename].nii.gz', the='name of the converted file')
class BiasCorrectionRunner(Module):
    """
    The N4 bias field correction algorithm is a popular method for correcting low frequency intensity 
    non-uniformity present in MRI image data known as a bias or gain field.
    """
    bundle_name: str                # TODO. make Optional[str] here and in decorator once supported
    converted_file_name: str

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path=IO.C('converted_file_name'), dtype='nifti:task=bias_corrected', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="converted data")
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

            # check datatype 
            if in_data.type.ftype == FileType.NIFTI:

                # N4 Bias Field Correction command
                n4_correction_command = [
                    "N4BiasFieldCorrection",
                    "-d",
                    "3",  # 3D image
                    "-i",
                    str(in_data.abspath),
                    "-o",
                    str(out_data.abspath),
                ]
                self.v("running N4BiasFieldCorrection....", n4_correction_command)
                # Run the N4 Bias Field Correction
                subprocess.run(n4_correction_command, check=True, capture_output=True)

            else:
                raise ValueError(f"CONVERT ERROR: unsupported file type {in_data.type.ftype}.")