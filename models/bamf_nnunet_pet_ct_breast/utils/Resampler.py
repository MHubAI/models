"""
-------------------------------------------------
MHub - Resampler Module
-------------------------------------------------

-------------------------------------------------
Author: Jithendra Kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from typing import List
from mhubio.core import Module, Instance, InstanceData, InstanceDataCollection, DataType, IO
import os, subprocess, shutil
import SimpleITK as sitk

@IO.ConfigInput('in_datas', 'nifti:mod=pt', the="list of input files to resample")
@IO.ConfigInput('in_target', 'nifti:mod=ct:registered=true', the="nifti reference data all must align to")
@IO.Config('bundle_name', str, None, the="bundle name converted data will be added to")
class Resampler(Module):

    source_segs: List[DataType]
    target_dicom: DataType
    bundle_name: str

    def resample_like(self, actual: sitk.Image, reference: sitk.Image, interpolator=sitk.sitkLinear) -> sitk.Image:
        """
        resample source image to reference. matches the spacing and dimension attributes to the reference image
        # use sitkNearestNeighbor as interpolator for changing a label image,
        # use sitkLinear for changing a scan image
        """
        filt = sitk.ResampleImageFilter()
        filt.SetReferenceImage(reference)
        filt.SetInterpolator(interpolator)
        converted = filt.Execute(actual)
        return converted

    @IO.Instance()
    @IO.Inputs('in_datas', the="list of input files to resample")
    @IO.Input('in_target', the="nifti reference data all must align to")
    @IO.Outputs('out_datas', path='u_[basename]', dtype='nifti:converted=true', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="converted data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, in_target: InstanceData, out_datas: InstanceDataCollection) -> None:

        assert isinstance(in_datas, InstanceDataCollection)
        assert isinstance(out_datas, InstanceDataCollection)
        assert len(in_datas) == len(out_datas)

        self.v(">> started: Resampling ")

        out_dir = self.config.data.requestTempDir(label="resampler-model-inp")
        ref_image = sitk.ReadImage(in_target.abspath)

        for i, in_data in enumerate(in_datas):
            out_data = out_datas.get(i)
            # check if output data already exists
            if os.path.isfile(out_data.abspath):
                print("CONVERT ERROR: File already exists: ", out_data.abspath)
                continue

            source_image = sitk.ReadImage(in_data.abspath)
            # get output data
            out_file = f'VOLUME_001_res.nii.gz'
            out_path = os.path.join(out_dir, out_file)
            conv_image = self.resample_like(source_image, ref_image, interpolator=sitk.sitkLinear)
            sitk.WriteImage(conv_image, out_path)
            # copy output data to instance
            shutil.copyfile(out_path, out_data.abspath)