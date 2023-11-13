"""
-------------------------------------------------
MedicalHub - Run Module for ensembling nnUNet inference.
-------------------------------------------------
-------------------------------------------------
Author: Rahul Soni
Email:  rahul.soni@bamfhealth.com
-------------------------------------------------
"""

from mhubio.core import Instance, InstanceData, DataType, FileType, CT, SEG
from mhubio.core import Module, IO
import os, numpy as np
import SimpleITK as sitk
from skimage import measure, filters
import numpy as np
import shutil


@IO.ConfigInput('in_data', 'nifti:mod=ct|mr', the="input data to run BamfProcessor on")
class BamfProcessorRunner(Module):

    @IO.Instance
    @IO.Input("in_data", the="input data to run nnunet on")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:

       # Log bamf runner info
        self.v("Running BamfProcessor on....")
        self.v(f" > input data:  {in_data.abspath}")
        self.v(f" > output data: {out_data.abspath}")

        # data
        # bring input data in nnunet specific format
        # NOTE: only for nifti data as we hardcode the nnunet-formatted-filename (and extension) for now.
        assert in_data.type.ftype == FileType.NIFTI
        assert in_data.abspath.endswith('.nii.gz')

        # read image
        self.v(f"Reading image from {in_data.abspath}")
        img_itk = sitk.ReadImage(in_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)

        # apply post-processing
        img_bamf_processed = self.n_connected(img_np)

        # store image temporarily
        out_file = os.path.join(instance.abspath, f'bamf_processed.nrrd')
        self.v(f"Writing tmp image to {out_file}")
        img_bamf_processed_itk = sitk.GetImageFromArray(img_bamf_processed)
        img_bamf_processed_itk.CopyInformation(img_itk)
        sitk.WriteImage(img_bamf_processed_itk, out_file)

        # meta
        meta = {
            "model": "BamfProcessor"
        }

        # create output data
        seg_data_type = DataType(FileType.NRRD, SEG + meta)
        seg_data = InstanceData(out_file, type=seg_data_type)
        instance.addData(seg_data)
        seg_data.confirm()


    def n_connected(self, img_data):
        img_data_mask = np.zeros(img_data.shape)
        img_data_mask[img_data > 0] = 1
        img_filtered = np.zeros(img_data_mask.shape)
        blobs_labels = measure.label(img_data_mask, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if count >= 1:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data
    
