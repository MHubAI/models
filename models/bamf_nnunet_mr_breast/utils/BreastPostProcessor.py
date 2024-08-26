"""
-------------------------------------------------
MHub - Run Module for ensembling nnUNet inference.
-------------------------------------------------
-------------------------------------------------
Author: Rahul Soni
Email:  rahul.soni@bamfhealth.com
-------------------------------------------------
"""

from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData
import os, shutil
import SimpleITK as sitk
import numpy as np
from skimage import measure, filters
from typing import Union
from pathlib import Path
import yaml



class BreastPostProcessor(Module):

    def get_mask(self, ip_path):
        # get segmentation mask
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
        seg_data[seg_data > 0] = 1
        return seg_data

    def n_connected(self, img_data: sitk.Image) -> sitk.Image:
        """
        Analyse connected components and drop smaller blobs
        """
        img_data_mask = np.zeros(img_data.shape)
        img_data_mask[img_data >= 1] = 1
        img_filtered = np.zeros(img_data_mask.shape)
        blobs_labels = measure.label(img_data_mask, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0
        for key, value in sorted_dict.items():
            if count >= 1 and count <= 2:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    def merge_segmentations(self, breast_and_fgt_seg:sitk.Image, tumor_seg:sitk.Image, mr_path:Union[str,Path]) -> sitk.Image:
        """
        Merge labels of breast, fgt, and breast tumor
        breast_and_fgt_seg: contains label=1 for breast, and label=2 for fgt
        tumor_seg: assign label=3 for breast tumor in the merge image
        returns: sitk image containing labels: [0,1,2,3]
        """
        tumor_seg[breast_and_fgt_seg == 0] = 0 # only focusing on predictions within the breast and fgt region
        breast_and_fgt_seg[tumor_seg == 1] = 3
        ref = sitk.ReadImage(mr_path)
        seg_img = sitk.GetImageFromArray(breast_and_fgt_seg)
        seg_img.CopyInformation(ref)
        return seg_img

    @IO.Instance()
    @IO.Input('in_breast_and_fgt_data', 'nifti:mod=seg:nnunet_dataset=Dataset009_Breast', the='input data from breast and fgt segmentation')
    @IO.Input('in_breast_tumor_data', 'nifti:mod=seg:nnunet_dataset=Dataset011_Breast', the='input data from breast tumor segmentation')
    @IO.Input('in_mr_data', 'nifti:mod=mr', the='input mr data')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=BREAST,BREAST+BREAST_FIBROGLANDULAR_TISSUE,BREAST+BREAST_CARCINOMA',
               the="get breast, fgt, and breast-tumor segmentation masks")
    def task(self, instance: Instance, in_breast_and_fgt_data: InstanceData, in_breast_tumor_data: InstanceData,
             in_mr_data: InstanceData, out_data: InstanceData):

        print('input breast and fgt segmentation: ' + in_breast_and_fgt_data.abspath)
        print('input breast tumor segmentation: ' + in_breast_tumor_data.abspath)
        print('output path: ' + out_data.abspath)

        # Breast and FGT segmentation
        breast_and_fgt_seg = self.get_mask(in_breast_and_fgt_data.abspath)
        breast_and_fgt_seg = self.n_connected(breast_and_fgt_seg)

        # Breast tumor segmentation
        tumor_seg = self.get_mask(in_breast_tumor_data.abspath)
        tumor_seg = self.n_connected(tumor_seg)

        # Merged segmentation masks
        output_seg = self.merge_segmentations(
            breast_and_fgt_seg = np.copy(breast_and_fgt_seg),
            tumor_seg = np.copy(tumor_seg),
            mr_path = in_mr_data.abspath
            )
        sitk.WriteImage(
            output_seg,
            out_data.abspath,
        )

