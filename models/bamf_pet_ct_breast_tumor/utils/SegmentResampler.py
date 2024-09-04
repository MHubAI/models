"""
---------------------------------------------------------
Resample Module using SimpleITK
---------------------------------------------------------

-------------------------------------------------
Author: Jithendra Kumar
Email:  Jithendra.kumar@bamfhealth.com
-------------------------------------------------

"""
import os
import shutil
import SimpleITK as sitk
import numpy as np
from mhubio.core import IO
from typing import List
from mhubio.core import Module, Instance, InstanceData, DataType, InstanceDataCollection


@IO.ConfigInput('in_datas', 'nifti:mod=SEG:roi=*', the="target segmentation files to resample ")
class SegmentResampler(Module):
    
    in_datas: List[DataType]

    @IO.Instance()
    @IO.Input('in_reference_data', 'nifti:mod=pt', the='input reference data')
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Input('in_transform', 'txt', the='transformation matrix data')        
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:resampled=true', data='in_datas',
                bundle='converted_tseg', auto_increment=True, the="resampled data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, in_reference_data: InstanceData,
             in_transform: InstanceData, out_datas: InstanceDataCollection):
        """
        Perform segment resampling using transform
        """
        self.v("Performing resampling...")
        self.v(f"Reference data: {in_reference_data.abspath}")

        # conversion step
        for i, in_data in enumerate(in_datas):
            out_data = out_datas.get(i)
            self.v(f"Input data: {in_data.abspath}")
            fixed = sitk.ReadImage(in_reference_data.abspath, sitk.sitkFloat32)
            moving = sitk.ReadImage(in_data.abspath, sitk.sitkFloat32)
            outTx = sitk.ReadTransform(in_transform.abspath)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(int(np.min(sitk.GetArrayFromImage(moving))))
            resampler.SetTransform(outTx)
            out = resampler.Execute(moving)
            out.CopyInformation(fixed)
            sitk.WriteImage(out, out_data.abspath)
