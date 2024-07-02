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
import SimpleITK as sitk
import numpy as np
import os, shutil
import cv2
from skimage import measure
from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData, InstanceDataCollection


class BreastPostProcessor(Module):

    def bbox2_3D(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    def n_connected(self, img_data):
        """
        Get the largest connected component in a binary image.

        Args:
            img_data (np.ndarray): image data.

        Returns:
            np.ndarray: Processed image with the largest connected component.
        """
        img_filtered = np.zeros(img_data.shape)
        blobs_labels = measure.label(img_data, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if count >= 1 and count <= 2 and value > 20:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    @IO.Instance()
    @IO.Input('in_ct_data', 'nifti:mod=ct:registered=true', the='input ct data')
    @IO.Input('in_tumor_data', 'nifti:mod=seg:model=nnunet', the='input tumor segmentation')
    @IO.Input('in_total_seg_data', 'nifti:mod=seg:model=TotalSegmentator', the='input total segmentation')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=BREAST+FDG_AVID_TUMOR', data='in_tumor_data',
               the="FDG-avid lesions in breast")
    def task(self, instance: Instance, in_ct_data: InstanceData, in_tumor_data: InstanceData,
             in_total_seg_data: InstanceData, out_data: InstanceData):
        """
        Perform postprocessing and writes simpleITK Image
        """
        tumor_seg_path = in_tumor_data.abspath
        total_seg_path = in_total_seg_data.abspath

        ts_data = sitk.GetArrayFromImage(sitk.ReadImage(total_seg_path))
        ts_abdominal = sitk.GetArrayFromImage(sitk.ReadImage(total_seg_path))
        ts_data[ts_data > 1] = 1
        lesions = sitk.GetArrayFromImage(sitk.ReadImage(tumor_seg_path))
        tumor_label = 9
        lesions[lesions != tumor_label] = 0
        lesions[lesions == tumor_label] = 1

        op_data = np.zeros(ts_data.shape)
        ref = sitk.ReadImage(in_ct_data.abspath)
        ct_data = sitk.GetArrayFromImage(ref)

        op_data[lesions == 1] = 1
        th = np.min(ct_data)
        op_data[ct_data == th] = 0  # removing predicitons where CT not available
        # Use the coordinates of the bounding box to crop the 3D numpy array.
        ts_abdominal[ts_abdominal > 4] = 0
        ts_abdominal[ts_abdominal > 1] = 1
        if ts_abdominal.max() > 0:
            x1, x2, y1, y2, z1, z2 = self.bbox2_3D(ts_abdominal)
        # Create a structuring element with ones in the middle and zeros around it
        structuring_element = np.ones((3, 3))

        # Dilate the array with the structuring element
        op_temp = cv2.dilate(ts_data, structuring_element, iterations=5)
        op_temp = cv2.erode(op_temp, structuring_element, iterations=5)
        op_data[op_temp == 1] = 0
        if ts_abdominal.max() > 0:
            op_data[x1:x2, y1:, :] = 0
        op_data[0:3, :, :] = 0
        op_data = self.n_connected(op_data)
        op_img = sitk.GetImageFromArray(op_data)
        op_img.CopyInformation(ref)
        tmp_dir = self.config.data.requestTempDir(label="breast-post-processor")
        tmp_file = os.path.join(tmp_dir, f'final.nii.gz')
        sitk.WriteImage(op_img, tmp_file)

        shutil.copyfile(tmp_file, out_data.abspath)
