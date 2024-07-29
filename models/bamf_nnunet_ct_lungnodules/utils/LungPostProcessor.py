"""
-------------------------------------------------
MHub - Run Module for perform postprocessing logic on segmentations.
-------------------------------------------------
-------------------------------------------------
Author: Jithendra Kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""
from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData
import SimpleITK as sitk
import numpy as np
from skimage import measure


class LungPostProcessor(Module):

    def perform_binary_threshold_segmentation(self, ip_path):
        """
        Perform binary threshold segmentation on the input image.

        Args:
        - ip_path (str): Path to the input image file.

        Returns:
        - numpy.ndarray: Segmented binary mask where non-zero values represent the segmented region.
        """
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
        seg_data[seg_data > 0] = 1
        return seg_data

    def extract_largest_connected_component(self, img_data):
        """
        Extract the single largest connected component from the segmentation image data.

        Args:
        - img_data (numpy.ndarray): Segmentation image data where connected components are to be identified.

        Returns:
        - numpy.ndarray: Binary image data with only the largest connected component retained.
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
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    def create_segmentation_image(self, lungs, nodules, ct_path):
        """
        Create a segmentation image combining lung and nodule segmentations.

        Args:
        - lungs (numpy.ndarray): Binary mask of lung segmentation.
        - nodules (numpy.ndarray): Binary mask of nodule segmentation.
        - ct_path (str): Path to the original CT image used as reference.

        Returns:
        - SimpleITK.Image: Segmentation image where lung and nodule regions are labeled as 1 and 2, respectively.
        """
        seg_data = np.zeros(lungs.shape)
        seg_data[lungs == 1] = 1
        seg_data[nodules == 1] = 2
        ref = sitk.ReadImage(ct_path)
        seg_img = sitk.GetImageFromArray(seg_data)
        seg_img.CopyInformation(ref)
        return seg_img

    @IO.Instance()
    @IO.Input('in_rg_data', 'nifti:mod=seg:nnunet_task=Task775_CT_NSCLC_RG', the='input data from lung nnunet module')
    @IO.Input('in_nodules_data', 'nifti:mod=seg:nnunet_task=Task777_CT_Nodules', the='input data from nodules nnunet nodule')
    @IO.Input('in_ct_data', 'nifti:mod=ct', the='input ct data')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=LUNG,LUNG+NODULE',
               data='in_rg_data', the="get lung and lung nodule segmentation file")
    def task(self, instance: Instance, in_rg_data: InstanceData, in_nodules_data: InstanceData,
             in_ct_data: InstanceData, out_data: InstanceData):

        self.v('running LungPostProcessor')

        seg_data = self.perform_binary_threshold_segmentation(in_rg_data.abspath)
        lungs = self.extract_largest_connected_component(seg_data)

        nodules = self.perform_binary_threshold_segmentation(in_nodules_data.abspath)
        nodules[lungs == 0] = 0

        final_seg_img = self.create_segmentation_image(np.copy(lungs), np.copy(nodules), in_ct_data.abspath)
        sitk.WriteImage(final_seg_img,out_data.abspath)
