"""
----------------------------------------------------------------
Mhub / DIAG - Run Module for ODE Colon Glands Segmentation model
----------------------------------------------------------------

----------------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
----------------------------------------------------------------
"""

from typing import List
from mhubio.core import Instance, DataTypeQuery, InstanceData, IO, Module

import os
import numpy as np
import SimpleITK as sitk

import torch
from PIL import Image

from ode_models import ConvODEUNet
from inference_utils import inference_image, postprocess



class OdeColonGlandsSegmentationRunner(Module):

    MODEL_WEIGHTS = "/opt/algorithm/best_border_unode_paper.pt"
    CACHED_MODEL = None

    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=wsi', the='input ct scan')
    @IO.Output('out_data', 'odecolonglands.mha', 'mha:mod=seg:model=OdeColonGlandsSegmentation', bundle='model', the='predicted segmentation of the colon glands')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        self.v(f"Reading image from {in_data.abspath}")
        img_itk = sitk.ReadImage(in_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)
        assert img_np.shape[2] == 3 and len(img_np.shape) == 3, f"Input image should have three dimensions with last dimension the number of colors (3), found: {img_np.shape}"
        img_pil = Image.fromarray(img_np)

        if self.CACHED_MODEL is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.v(f"Loading model from {self.MODEL_WEIGHTS} on device: {device}")
            net = ConvODEUNet(num_filters=16, output_dim=2, time_dependent=True,
                              non_linearity='lrelu', adjoint=True, tol=1e-3)
            net.to(device)
            net.load_state_dict(torch.load(self.MODEL_WEIGHTS, map_location=device))
            self.CACHED_MODEL = (net, device)
        else:
            net, device = self.CACHED_MODEL

        self.v("Applying ODE Colon Glands segmentation")
        seg_result_np, _ = inference_image(net, img_pil, shouldpad=False)
        self.v(seg_result_np.shape)
        seg_result_np = postprocess(seg_result_np, img_pil)
        self.v(seg_result_np.shape)

        self.v(f"Writing image to {out_data.abspath}")
        seg_itk = sitk.GetImageFromArray(seg_result_np)
        if len(img_itk.GetSize()) == 3:
            seg_itk.CopyInformation(img_itk[0, :, :])
        else:
            seg_itk.CopyInformation(img_itk)
        sitk.WriteImage(seg_itk, out_data.abspath)
