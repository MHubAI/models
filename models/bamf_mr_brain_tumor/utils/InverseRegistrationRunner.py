"""
-------------------------------------------------
MHub - Inverse Registration Module to t1c mr brain
-------------------------------------------------

-------------------------------------------------
Author: Jithendra
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from typing import List
from mhubio.core import Module, Instance, InstanceDataCollection, InstanceData, DataType, FileType
from mhubio.core.IO import IO
import SimpleITK as sitk

import os, subprocess, uuid

@IO.ConfigInput('in_seg_datas', 'nifti', the="data to be converted")
@IO.ConfigInput('in_mat_datas',  'txt', the="transformation matrix data")
@IO.ConfigInput('in_registration_datas', 'nifti:mod=mr', the="registered data")
class InverseRegistrationRunner(Module):
    """
    # Inverse registration using FLIRT
    """
    in_seg_datas: List[DataType]
    in_mat_datas: List[DataType]
    in_registration_datas: List[DataType]    

    @IO.Instance()
    @IO.Inputs('in_seg_datas', the="data to be converted")
    @IO.Inputs('in_mat_datas', the="transformation matrix data")
    @IO.Inputs('in_registration_datas', the="registered data")
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:task=inverse:roi=NECROSIS,EDEMA,ENHANCING_LESION',
                data='in_registration_datas', bundle='inverse_t1c_registration', auto_increment=True, the="converted data")
    def task(self, instance: Instance, in_seg_datas : InstanceDataCollection, in_mat_datas: InstanceDataCollection, in_registration_datas: InstanceDataCollection,  out_datas: InstanceDataCollection, **kwargs) -> None:

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
            in_seg_data = in_seg_datas.get(i)

            # check datatype 
            reverse_transformation_matrix = os.path.join(process_dir, f'{str(uuid.uuid4())}_reverse.mat')
            reverse_transformation_file = os.path.join(process_dir, f'{str(uuid.uuid4())}.nii.gz')
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
                self.subprocess(convert_command, check=True)
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
                str(reverse_transformation_file),
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
            self.subprocess(cmd, check=True)
            # Load your image
            image = sitk.ReadImage(reverse_transformation_file)

            # Change the label from 4 to 3
            label_map = {4: 3}
            changed_image = sitk.ChangeLabel(image, changeMap=label_map)

            # Add the converted data to the output collection
            sitk.WriteImage(changed_image, out_data.abspath)