"""
-------------------------------------------------
MHub - FLIRT Registration Module
-------------------------------------------------

-------------------------------------------------
Author: Jithendra
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from typing import List
from pathlib import Path
from mhubio.core import Module, Instance, InstanceDataCollection, InstanceData, DataType, FileType
from mhubio.core.IO import IO

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be registered")
@IO.ConfigInput('reference_data', 'nifti:mod=mr', the="reference data all segmentations register to")
@IO.Config('degrees_of_freedom', str, '6', the='degrees of freedom for registration')
class FLIRTRegistrationRunner(Module):
    """
    # Rigid registration using FLIRT
    """
    in_datas: List[DataType]
    reference_data: DataType
    degrees_of_freedom: str

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Input('reference_data', the="reference data all segmentations register to")
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:task=registration',
                data='in_datas', bundle='t1c_registration', auto_increment=True, the="converted data")
    @IO.Outputs('out_mat_datas', path='[filename]_transform_mat.txt', dtype='txt:task=registration_transform_mat',
                data='in_datas', bundle='t1c_registration', auto_increment=True, the="transformation matrix data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, reference_data : InstanceData, out_datas: InstanceDataCollection, out_mat_datas: InstanceDataCollection, **kwargs) -> None:
        """
        6 Degrees of Freedom (Rigid Registration)
        Description: Allows for translation and rotation.
        Parameters Controlled:
            3 for translation (x, y, z)
            3 for rotation (pitch, yaw, roll)
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

            reference_path = reference_data.abspath

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
                self.degrees_of_freedom,  # 6 degrees of freedom for rigid registration
            ]
            self.v("running FLIRT....", cmd)
            self.subprocess(cmd, check=True)
