"""
-------------------------------------------------
MHub - FLIRT Registration Module to standard atlas t1 brain
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

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be registered")
@IO.Config('bundle_name', str, 'atlas_registration', the="bundle name converted data will be added to")
@IO.Config('converted_file_name', str, '[filename].nii.gz', the='name of the converted file')
@IO.Config('transformation_file_name', str, '[filename]_transform_mat.txt', the='name of the transformation matrix file')
class StdRegistrationRunner(Module):
    """
    # Rigid registration using FLIRT
    """
    in_datas: List[DataType]
    bundle_name: str                # TODO. make Optional[str] here and in decorator once supported
    converted_file_name: str
    transformation_file_name: str

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path=IO.C('converted_file_name'), dtype='nifti:task=std_registration', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="converted data")
    @IO.Outputs('out_mat_datas', path=IO.C('transformation_file_name'), dtype='txt:task=std_registration_transform_mat', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="transformation matrix data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, out_datas: InstanceDataCollection, out_mat_datas: InstanceDataCollection, **kwargs) -> None:
        """
        12 Degrees of Freedom (Affine Registration)
        Description: Allows for translation, rotation, scaling, and shearing.
        Parameters Controlled:
            3 for translation (x, y, z)
            3 for rotation (pitch, yaw, roll)
            3 for scaling (scale in x, y, z directions)
            3 for shearing (xy, xz, yz)
        """

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
            out_mat_data = out_mat_datas.get(i)
            reference_path = 'models/bamf_mr_brain_tumor/utils/templates/T1_brain.nii'

            # check datatype 
            if in_data.type.ftype == FileType.NIFTI:
                cmd = [
                    "flirt",
                    "-in",
                    str(in_data.abspath),
                    "-ref",
                    str(reference_path),
                    "-out",
                    str(out_data.abspath),
                    "-omat",
                    str(out_mat_data.abspath),  # Save the transformation matrix
                    "-dof",
                    "12", 
                ]
                self.v(f" > bash_command:     {cmd}")
                subprocess.run(cmd, check=True)
            else:
                raise ValueError(f"CONVERT ERROR: unsupported file type {in_data.type.ftype}.")