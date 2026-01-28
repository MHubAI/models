"""
-------------------------------------------------
MHub - Run Module for post processing on segmentations
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

class PostProcessor(Module):

    def n_connected(self, img_data: np.ndarray) -> np.ndarray:
        """
        Filters the connected components in the image, retaining only the largest components.

        Parameters:
        - img_data (np.ndarray): The input binary image.

        Returns:
        - np.ndarray: The filtered binary image.
        """
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
            if count == 1:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=seg:model=nnunet', the='input segmentations')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=LIVER,LIVER+NEOPLASM', data='in_data', the="filtered Liver and tumor segmentation")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:

        # Log bamf runner info
        self.log("Running Segmentation Post Processing on....")
        self.log(f" > input data:  {in_data.abspath}")
        self.log(f" > output data: {out_data.abspath}")

        label_img = sitk.ReadImage(in_data.abspath)
        seg_data = sitk.GetArrayFromImage(label_img)
        seg_data[seg_data < 8] = 0
        seg_data[seg_data == 8] = 1
        seg_data[seg_data == 9] = 2
        seg_data = self.n_connected(seg_data)
        filtered_label_img = sitk.GetImageFromArray(seg_data)
        filtered_label_img.CopyInformation(label_img)
        sitk.WriteImage(filtered_label_img, out_data.abspath)