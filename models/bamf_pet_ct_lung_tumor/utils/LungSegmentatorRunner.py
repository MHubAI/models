"""
-------------------------------------------------
MHub - Run Lung segmentator Module using TotalSegmentator.
-------------------------------------------------

-------------------------------------------------
Author: Jithendra Kumar
Email:  Jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, FileType, CT, SEG, IO, DataTypeQuery
import os, subprocess
import SimpleITK as sitk
import numpy as np
from skimage import measure
from mhubio.core import IO

from totalsegmentator.map_to_binary import class_map


@IO.ConfigInput('in_data', 'nifti:mod=ct', the="input data to run Lung Segmentator on")
class LungSegmentatorRunner(Module):

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

    @IO.Instance()
    @IO.Input('in_data', the="input whole body ct scan")
    @IO.Output('out_data', 'lung_segmentations.nii.gz', 'nifti:mod=seg:model=LungSegmentator:roi=LEFT_LUNG,RIGHT_LUNG',
               data='in_data', the="output segmentation mask containing lung labels")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        # use total segmentator to extract lung segmentation
        bash_command = ["TotalSegmentator"]
        bash_command += ["-i", in_data.abspath]

        tmp_dir = self.config.data.requestTempDir(label="lung-segment-processor")
        segments_file = os.path.join(tmp_dir, f'segmentations.nii.gz')

        # multi-label output (one nifti file containing all labels instead of one nifti file per label)
        self.v("Generating multi-label output ('--ml')")
        bash_command += ["-o", segments_file]
        bash_command += ["--ml"]

        # fast mode
        self.v("Running TotalSegmentator in default mode (1.5mm)")
        self.v(">> run: ", " ".join(bash_command))

        # run the model
        self.subprocess(bash_command, text=True)

        # Extract labels for left lung and right lung from total segmentator v1 output
        left_lung_labels = [label for label, name in class_map["total"].items() if "left" in name and "lung" in name]
        right_lung_labels = [label for label, name in class_map["total"].items() if "right" in name and "lung" in name]

        segments_arr = sitk.GetArrayFromImage(sitk.ReadImage(segments_file))
        lung_left = self.n_connected(self.mask_labels(left_lung_labels, segments_arr))
        lung_right = self.n_connected(self.mask_labels(right_lung_labels, segments_arr))
        op_data = np.zeros(segments_arr.shape)
        op_data[lung_left > 0] = 1
        op_data[lung_right > 0] = 2
        ref = sitk.ReadImage(in_data.abspath)
        op_img = sitk.GetImageFromArray(op_data)
        op_img.CopyInformation(ref)
        sitk.WriteImage(op_img, out_data.abspath)