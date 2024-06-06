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


class BreastPostProcessor(Module):

    def get_mask(self, ip_path):
        # get segmentation mask
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
        seg_data[seg_data > 0] = 1
        return seg_data

    def n_connected(self, img_data):
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

    def n4_bias_correction(
        input_image,
        shrink_factor=1,
        mask_image=None,
        number_of_iterations=50,
        number_of_fitting_levels=4,
    ):
        if mask_image is None:
            mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)

        if shrink_factor > 1:
            input_image = sitk.Shrink(
                input_image, [shrink_factor] * input_image.GetDimension()
            )
            mask_image = sitk.Shrink(
                mask_image, [shrink_factor] * mask_image.GetDimension()
            )

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            [number_of_iterations] * number_of_fitting_levels
        )

        corrector.Execute(input_image, mask_image)
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
        corrected_image_full_resolution = input_image / sitk.Exp(log_bias_field)
        return corrected_image_full_resolution

    def get_seg_img(self, breast_and_fgt, breast_tumor, mr_path):
        seg_data = np.zeros(breast_and_fgt.shape)
        seg_data[breast_and_fgt == 1] = 1
        seg_data[breast_tumor == 1] = 2
        ref = sitk.ReadImage(mr_path)
        seg_img = sitk.GetImageFromArray(seg_data)
        seg_img.CopyInformation(ref)
        return seg_img


    @IO.Instance()
    @IO.Input('in_breast_and_fgt_data', 'nifti:mod=seg:nnunet_task=Dataset009_Breast', the='input data from breast and fgt segmentation')
    @IO.Input('in_breast_tumor_data', 'nifti:mod=seg:nnunet_task=Dataset011_Breast', the='input data from breast tumor segmentation')
    @IO.Input('in_mr_data', 'nifti:mod=mr', the='input mr data')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=BREAST,BREAST+FGT,BREAST_TUMOR',
               the="get breast_fgt and breast segmentation file")
    def task(self, instance: Instance, in_breast_and_fgt_data: InstanceData, in_breast_tumor_data: InstanceData,
             in_mr_data: InstanceData, out_data: InstanceData):

        print('input breast and fgt segmentation: ' + in_breast_and_fgt_data.abspath)
        print('input breast tumor segmentation: ' + in_breast_tumor_data.abspath)
        print('output path: ' + out_data.abspath)

        breast_fgt_seg = self.get_mask(in_breast_and_fgt_data.abspath)
        breast_fgt_seg = self.n_connected(breast_fgt_seg)

        breast_tumor_seg = self.get_mask(in_breast_tumor_data.abspath)
        breast_tumor_seg[breast_fgt_seg == 0] = 0

        seg_img = self.get_seg_img(
            breast_and_fgt = np.copy(breast_fgt_seg),
            breast_tumor = np.copy(breast_tumor_seg),
            mr_path = in_mr_data.abspath
            )
        process_dir = self.config.data.requestTempDir(label="nnunet-breast-processor")
        process_file = os.path.join(process_dir, f'final.nii.gz')
        sitk.WriteImage(
            seg_img,
            process_file,
        )
        shutil.copyfile(process_file, out_data.abspath)