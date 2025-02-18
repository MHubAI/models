"""
-------------------------------------------------
MHub - Run Module for perform postprocessing logic on segmentations.
-------------------------------------------------
-------------------------------------------------
Author: Jithendra Kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from mhubio.core import Instance, InstanceData
from mhubio.core import Module, IO
from skimage import measure
import SimpleITK as sitk
import numpy as np


class BamfProcessorRunner(Module):

    def max_planar_dimension(self, label_img, label_cnt):
        """
        Calculate the maximum planar dimension of a specific label in a 3D label image.

        Args:
            label_img (sitk.Image): The 3D label image.
            label_cnt (int): The label number to analyze.

        Returns:
            float: The maximum size of the label in millimeters (mm) across the most planar dimension.
        """
        tumor = label_img == label_cnt

        assert tumor.GetDimension() == 3
        spacing = tumor.GetSpacing()
        if spacing[0] == spacing[1] and spacing[1] != spacing[2]:
            axis = 2
            plane_space = spacing[0]
        elif spacing[0] != spacing[1] and spacing[1] == spacing[2]:
            axis = 0
            plane_space = spacing[1]
        else:
            axis = 1
            plane_space = spacing[2]

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(tumor)

        boundingBox = np.array(lsif.GetBoundingBox(1))
        sizes = boundingBox[3:].tolist()
        del sizes[axis]
        max_planar_size = plane_space * max(sizes)  # mm
        return max_planar_size

    def filter_nodules(self, label_img, min_size=3):
        """
        Filter lung nodules based on their size and re-label them accordingly.

        Args:
            label_img (sitk.Image): The 3D label image containing lung and nodule labels.
            min_size (float): Minimum planar size (in mm) to retain a nodule.

        Returns:
            sitk.Image: The processed label image with nodules filtered by size.
        """
        label_val_lung = 1
        label_val_nodule = 2
        label_val_large_nodule = 3

        nodules_img = label_img == label_val_nodule
        nodule_components = sitk.ConnectedComponent(nodules_img)

        nodules_to_remove = []

        for lbl in range(1, sitk.GetArrayFromImage(nodule_components).max() + 1):
            max_size = self.max_planar_dimension(nodule_components, lbl)

            if max_size < min_size:
                nodules_to_remove.append(lbl)
                # print("Removing label", lbl, "with size", max_size)
            elif 3 <= max_size <= 30:
                label_img = sitk.ChangeLabel(label_img, {lbl: label_val_nodule})
                # print("Marking label", lbl, "as Nodule (label 2) with size", max_size)
            else:
                label_img = sitk.ChangeLabel(label_img, {lbl: label_val_large_nodule})
                # print("Marking label", lbl, "as Large Nodule (label 3) with size", max_size)

        label_img = sitk.ChangeLabel(label_img, {label_val_nodule: label_val_lung})
        big_nodules = sitk.ChangeLabel(nodule_components, {x: 0 for x in nodules_to_remove})
        label_img = sitk.Mask(label_img, big_nodules > 0, label_val_nodule, label_val_lung)
        label_img = self.n_connected(label_img)

        return label_img


    def n_connected(self, img):
        """
        Retain the largest connected components in a binary label image.

        Args:
            img (sitk.Image): The input binary label image.

        Returns:
            sitk.Image: The processed image with only the largest connected components retained.
        """
        img_data = sitk.GetArrayFromImage(img)
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
            if count >= 1 and count <= 2:
                if count == 1:
                    val = value
                    img_filtered[blobs_labels == key] = 1
                if count == 2 and value > (val * 0.2):
                    img_filtered[blobs_labels == key] = 1

            count += 1

        img_data[img_filtered != 1] = 0
        img_masked = sitk.GetImageFromArray(img_data)
        img_masked.CopyInformation(img)
        return img_masked

    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=seg:model=nnunet', the='input segmentations')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=LUNG,LUNG+NODULE', data='in_data', the="lung and filtered nodules segmentation")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        """
        Main task function that processes the input lung and nodule segmentations,
        filters nodules based on their size, and writes the output image.

        Args:
            instance (Instance): The MHub instance for processing.
            in_data (InstanceData): Input data containing the segmentation.
            out_data (InstanceData): Output data path to save the processed image.
        """
        # Log bamf runner info
        self.log("Running BamfProcessor on....")
        self.log(f" > input data:  {in_data.abspath}")
        self.log(f" > output data: {out_data.abspath}")

        label_img = sitk.ReadImage(in_data.abspath)
        filtered_label_img = self.filter_nodules(label_img, min_size=3)
        sitk.WriteImage(filtered_label_img, out_data.abspath)