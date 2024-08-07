"""
-------------------------------------------------
MHub - Inverse Registration Module to atlas t1 brain
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

import os, subprocess, uuid


@IO.ConfigInput('in_seg_data', 'nifti:mod=mr', the="data to be converted")
@IO.ConfigInput('in_mat_datas', 'txt', the="transformation matrix data")
@IO.ConfigInput('in_registration_datas', 'nifti:mod=mr', the="registered data")
@IO.Config('bundle_name', str, 'inverse_atlas_registration', the="bundle name converted data will be added to")
@IO.Config('converted_file_name', str, '[filename].nii.gz', the='name of the converted file')
class InverseStdRegistrationRunner(Module):
    """
    # Inverse registration using FLIRT
    """
    in_seg_data: DataType
    in_mat_datas: List[DataType]
    in_registration_datas: List[DataType]    
    bundle_name: str                # TODO. make Optional[str] here and in decorator once supported
    converted_file_name: str

    @IO.Instance()
    @IO.Input('in_seg_data', the="data to be converted")
    @IO.Inputs('in_mat_datas', the="transformation matrix data")
    @IO.Inputs('in_registration_datas', the="registered data")
    @IO.Outputs('out_datas', path=IO.C('converted_file_name'), dtype='nifti:task=std_inverse', data='in_mat_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="converted data")
    def task(self, instance: Instance, in_seg_data : InstanceData,in_mat_datas: InstanceDataCollection, in_registration_datas: InstanceDataCollection,  out_datas: InstanceDataCollection, **kwargs) -> None:

        # some sanity checks
        assert isinstance(in_registration_datas, InstanceDataCollection)
        assert isinstance(out_datas, InstanceDataCollection)
        assert len(in_registration_datas) == len(out_datas)

        # filtered collection must not be empty
        if len(in_registration_datas) == 0:
            self.v(f"CONVERT ERROR: no data found in instance {str(instance)}.")
            return None

        process_dir = self.config.data.requestTempDir(label="inverse-processor")

        # conversion step
        for i, in_data in enumerate(in_registration_datas):
            in_mat = in_mat_datas.get(i)
            out_data = out_datas.get(i)

            # check datatype 
            if in_data.type.ftype == FileType.NIFTI:                
                reverse_transformation_matrix = os.path.join(process_dir, f'{str(uuid.uuid4())}_reverse.mat') 
                # Command to convert the transformation matrices
                convert_command = [
                    "convert_xfm",
                    "-omat",
                    str(reverse_transformation_matrix),
                    "-inverse",
                    str(in_mat.abspath),
                ]
                
                try:
                    self.v("Converting transformation matrices...",convert_command)
                    subprocess.run(convert_command, check=True)
                    self.v("Transformation matrices converted successfully.")
                except subprocess.CalledProcessError as e:
                    self.v("Error converting transformation matrices:", e)
                # Inverse registration using FLIRT with the saved transformation matrix
                cmd = [
                    "flirt",
                    "-in",
                    str(in_seg_data.abspath),
                    "-ref",
                    str(in_data.abspath),
                    "-out",
                    str(out_data.abspath),
                    "-init",
                    str(
                        reverse_transformation_matrix
                    ),  # Use the saved transformation matrix for inverse registration
                    "-cost",
                    "normmi",
                    "-dof",
                    "12",  # 6 degrees of freedom for rigid registration
                    "-interp",
                    "nearestneighbour",  # Interpolation method (adjust as needed)
                    "-applyxfm",
                ]
                self.v("inverse transformation flirt...",cmd)
                subprocess.run(cmd, check=True)                
            else:
                raise ValueError(f"CONVERT ERROR: unsupported file type {in_data.type.ftype}.")