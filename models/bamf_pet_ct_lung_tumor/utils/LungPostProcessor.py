"""
---------------------------------------------------------
Post processing Module on segmentation output
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
from skimage import measure
from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData, InstanceDataCollection

     
class LungPostProcessor(Module):
    
    def mask_labels(self, labels, ts):
        """
        Create a mask based on given labels.

        Args:
            labels (list): List of labels to be masked.
            ts (np.ndarray): Image data.

        Returns:
            np.ndarray: Masked image data.
        """
        lung = np.zeros(ts.shape)
        for lbl in labels:
            lung[ts == lbl] = 1
        return lung

    def n_connected(self, img_data):
        """
        Get the largest connected component in a binary image.

        Args:
            img_data (np.ndarray): image data.

        Returns:
            np.ndarray: Processed image with the largest connected component.
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
            if count == 1:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    def arr_2_sitk_img(self, arr, ref):
        """
        Convert numpy array to SimpleITK image.

        Args:
            arr (np.ndarray): Input image data as a numpy array.
            ref: Reference image for copying information.

        Returns:
            sitk.Image: Converted SimpleITK image.
        """
        op_img = sitk.GetImageFromArray(arr)
        op_img.CopyInformation(ref)
        return op_img

    def get_mets(self, left, op_data):
        """
        Perform metastasis segmentation.

        Args:
            left (np.ndarray): Image data for left lung.
            op_data (np.ndarray): Image data for segmented regions.

        Returns:
            np.ndarray: Metastasis segmentation results.
        """
        op_data[left == 1] = 0
        op_primary = self.n_connected(np.copy(op_data))
        mets = np.zeros(op_primary.shape)
        mets[op_data > 0] = 1
        mets[op_primary > 0] = 0
        return mets
    
    def get_lung_ts(self, img_path):
        """
        Perform lung tissue segmentation.

        Args:
            img_path (str): Path to the image for lung tissue segmentation.

        Returns:
            tuple: A tuple containing lung segmentation results.
        """
        img_data = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        left_labels = [13, 14]  # defined in totalsegmentator
        right_labels = [15, 16, 17]  # defined in totalsegmentator
        heart_labels = [44, 45, 46, 47, 48]  # defined in totalsegmentator
        lung_left = self.n_connected(self.mask_labels(left_labels, img_data))
        lung_right = self.n_connected(self.mask_labels(right_labels, img_data))
        heart = self.n_connected(self.mask_labels(heart_labels, img_data))
        return lung_left, lung_right, lung_right + lung_left, heart
    
    @IO.Instance()
    @IO.Input('in_ct_data', 'nifti:mod=ct:registered=true', the='input ct data')
    @IO.Input('in_tumor_data', 'nifti:mod=seg:model=nnunet', the='input tumor segmentation')
    @IO.Input('in_total_seg_data', 'nifti:mod=seg:model=TotalSegmentator', the='input total segmentation')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=LUNG,LUNG+FDG_AVID_TUMOR',
               data='in_tumor_data',
               the="get the lung and tumor after post processing")
    def task(self, instance: Instance, in_ct_data: InstanceData, in_tumor_data: InstanceData,
             in_total_seg_data: InstanceData, out_data: InstanceData):
        """
        Perform postprocessing and writes simpleITK Image
        """
        self.v("Running LungPostprocessor.")
        tumor_seg_path = in_tumor_data.abspath
        total_seg_path = in_total_seg_data.abspath

        right, left, lung, heart = self.get_lung_ts(str(total_seg_path))
        tumor_label = 9
        tumor_arr = sitk.GetArrayFromImage(sitk.ReadImage(tumor_seg_path))
        tumor_arr[tumor_arr != tumor_label] = 0

        op_data = np.zeros(lung.shape)
        ref = sitk.ReadImage(in_ct_data.abspath)
        ct_data = sitk.GetArrayFromImage(ref)
        op_data[lung == 1] = 1
        op_data[tumor_arr > 0] = 2
        th = np.min(ct_data)
        op_data[ct_data == th] = 0  # removing predictions where CT not available
        mets_right = self.get_mets(left, np.copy(op_data))
        mets_left = self.get_mets(right, np.copy(op_data))
        mets = np.logical_and(mets_right, mets_left).astype("int")
        op_data[mets == 1] = 3
        op_data[op_data == 3] = 0

        op_img = sitk.GetImageFromArray(op_data)
        op_img.CopyInformation(ref)

        tmp_dir = self.config.data.requestTempDir(label="lung-post-processor")
        tmp_file = os.path.join(tmp_dir, f'final.nii.gz')
        sitk.WriteImage(op_img, tmp_file)
        shutil.copyfile(tmp_file, out_data.abspath)