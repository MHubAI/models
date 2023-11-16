"""
-------------------------------------------------
MHub - Run Module for ensembling nnUNet inference.
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



class BamfProcessorRunner(Module):

    @IO.Instance
    @IO.Input('in_data', 'nifti:mod=ct|mr', the='input data to run nnunet on')
    @IO.Output('out_data', 'nrrd:mod=seg:processor=bamf', data='in_data', the="keep the two largest connected components of the segmentation and remove all other ones")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
       # Log bamf runner info
        self.v("Running BamfProcessor on....")
        self.v(f" > input data:  {in_data.abspath}")
        self.v(f" > output data: {out_data.abspath}")

        # read image
        self.v(f"Reading image from {in_data.abspath}")
        img_itk = sitk.ReadImage(in_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)

        # apply post-processing
        img_bamf_processed = self.n_connected(img_np)

        # store image temporarily
        self.v(f"Writing tmp image to {out_data.abspath}")
        img_bamf_processed_itk = sitk.GetImageFromArray(img_bamf_processed)
        img_bamf_processed_itk.CopyInformation(img_itk)
        sitk.WriteImage(img_bamf_processed_itk, out_data.abspath)


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
