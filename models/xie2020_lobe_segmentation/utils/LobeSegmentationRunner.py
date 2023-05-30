"""
------------------------------------------------------
Mhub / DIAG - Run Module for Xie2020 Lobe Segmentation
------------------------------------------------------

------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------
"""

from mhubio.core import Instance, InstanceData, IO, Module

import os
import numpy as np
import SimpleITK as sitk

from test import segment_lobe, segment_lobe_init

class LobeSegmentationRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'nrrd:mod=ct', the='input ct scan')
    @IO.Output('out_data', 'xie2020lobeseg.nrrd', 'nrrd:mod=seg:model=Xie2020LobeSegmentation', 'in_data', the='predicted segmentation of the lung lobes')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        # NOTE input data originally was specified for MHA/MHD and could be extended for DICOM

        # read image
        self.v(f"Reading image from {in_data.abspath}")
        img_itk = sitk.ReadImage(in_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)

        # apply lobe segmentation
        origin = img_itk.GetOrigin()[::-1]
        spacing = img_itk.GetSpacing()[::-1]
        direction = np.asarray(img_itk.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        meta_dict =  {
            "uid": os.path.basename(in_data.abspath),
            "size": img_np.shape,
            "spacing": spacing,
            "origin": origin,
            "original_spacing": spacing,
            "original_size": img_np.shape,
            "direction": direction
        }

        handle = segment_lobe_init()
        seg_result_np = segment_lobe(handle, img_np, meta_dict)

        # store image
        self.v(f"Writing image to {out_data.abspath}")
        seg_itk = sitk.GetImageFromArray(seg_result_np)
        seg_itk.CopyInformation(img_itk)
        sitk.WriteImage(seg_itk, out_data.abspath)