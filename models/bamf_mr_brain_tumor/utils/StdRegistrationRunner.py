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

REFERENCE_PATH = Path(__file__).parent.parent / "src" / "templates" / "T1_brain.nii"

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be registered")
class StdRegistrationRunner(Module):
    """
    # Rigid registration using FLIRT
    """
    in_datas: List[DataType]

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:task=std_registration',
                data='in_datas', bundle='atlas_registration', auto_increment=True, the="converted data")
    @IO.Outputs('out_mat_datas', path='[filename]_transform_mat.txt', dtype='txt:task=std_registration_transform_mat',
                data='in_datas', bundle='atlas_registration', auto_increment=True, the="transformation matrix data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, out_datas: InstanceDataCollection,
             out_mat_datas: InstanceDataCollection, **kwargs) -> None:
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

            cmd = [
                "flirt",
                "-in",
                str(in_data.abspath),
                "-ref",
                str(REFERENCE_PATH),
                "-out",
                str(out_data.abspath),
                "-omat",
                str(out_mat_data.abspath),  # Save the transformation matrix
                "-dof",
                "12",
            ]
            self.v(f" > bash_command:     {cmd}")
            self.subprocess(cmd, text=True)
