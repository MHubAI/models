"""
-------------------------------------------------
DeepCAC2 - 3D Unet Pipeline

This script runs the entrire processing pieline based on nrrd input chest ct scans:
(loading data, preprocessing, patching, model execution, reassembling, ags computation)
on all patients from the validation split from start to finish based on a
pre-trained 3D Unet (see model.py).
Therefore, this script is INDEPENDENT from any preparations (see preparedata.py) 
which is used soley for training speedup.

Metrics can then be calculated in the next step based on the predicted and assembled cac segmentation.

WORK IN PROGRESS.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

# imports
from typing import Tuple

import os
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio

from .model import UNET3D
from .subject import Sample, get_subject, get_tfx

# alpha
cWEIGHTS = 'vivid-haze-6_model.pt'
cCOMMON_SPACING = 0.7, 0.7, 2.5

# beta
#cWEIGHTS = 'glad-fire-13_model.pt'
#cCOMMON_SPACING = 0.68, 0.68, 2.5

cDEVICE = 'cuda:0'
cBOUNDING_BOX = 256, 256, 58
cPATCHSIZE = 64, 64, 16
cSTRIDE = 32, 32, 8
cTH = 0.5
cAPPLY_ZNORM = True

# instantiate model & load trained weights 
model = UNET3D(1, 1)
model.to(cDEVICE)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'weights', cWEIGHTS)))
model.eval()

# 
def predict_sample(
        sample: Sample, 
        tfx: tio.Transform,
        patchsize: Tuple[int, int, int], 
        stride: Tuple[int, int, int],
        threshold: float,
        predcac_seg_file: str
    ):

    # load subject
    subject = get_subject(sample)

    # preprocess subject
    subject_tfx = tfx(subject)
    assert hasattr(subject_tfx, 'image') and hasattr(subject_tfx, 'heart')

    # extract data as numpy arrays
    image_np = subject_tfx.image.numpy().squeeze() # type: ignore
    heart_np = subject_tfx.heart.numpy().squeeze() # type: ignore

    # 
    img_vol = sitk.ReadImage(sample['img_path'])

    # get shape dim
    w, h, d = image_np.shape
    pw, ph, pd = patchsize

    # calculate / estimate the number of patches
    num_patches = 0
    for wi in range(0, w-pw, stride[0]):
        for hi in range(0, h-ph, stride[1]):
            for di in range(0, d-pd, stride[2]):           
                heart_patch = heart_np[wi:wi+pw,hi:hi+ph,di:di+pd]
                assert heart_patch.shape == patchsize

                # ignore patches with no heart present
                if heart_patch.sum() > 0:
                    num_patches += 1

    out = np.zeros((w, h, d, num_patches))
    div = np.zeros((w, h, d))

    patch_i = 0
    for wi in range(0, w-pw, stride[0]):
        for hi in range(0, h-ph, stride[1]):
            for di in range(0, d-pd, stride[2]):
                image_patch = image_np[wi:wi+pw,hi:hi+ph,di:di+pd]
                heart_patch = heart_np[wi:wi+pw,hi:hi+ph,di:di+pd]
                assert image_patch.shape == patchsize

                # ignore patches with no heart present
                if heart_patch.sum() == 0:
                    continue

                ins = torch.tensor(np.array(image_patch)).to(cDEVICE).unsqueeze(0).unsqueeze(1)
                pred = model(ins).sigmoid().squeeze(1).squeeze(0).detach().cpu().numpy()

                out[wi:wi+pw,hi:hi+ph,di:di+pd,patch_i] = pred
                div[wi:wi+pw,hi:hi+ph,di:di+pd] += 1

                patch_i += 1

    div[div == 0] = 1

    out = np.sum(out, axis=3) / div 

    out_final = (out > threshold).astype(int)

    # add the prediction to the transformed subject
    subject_tfx['cacpred'] = tio.LabelMap(tensor=torch.Tensor(out_final).unsqueeze(0), affine=subject_tfx.image.affine) # type: ignore

    # inverse (for some reason affien is ignored by apply_inverse_transform although it shoul dbe invertible?!)
    resample_itfx = tio.Resample(subject.image) # type: ignore

    # inverse pre-processing transformations on the subject
    subject_tt = resample_itfx(subject_tfx.apply_inverse_transform(image_interpolation='linear')) # type: ignore

    # again etract an numpy array from the subject and pass it to sitk
    # NOTE sitk expects data in z, y, x orientation (thus transpose)!
    sitk_vol = sitk.GetImageFromArray(subject_tt['cacpred'].numpy().squeeze(0).transpose(2, 1, 0)) # type: ignore

    # now load the original image using sitk and apply all meta-data (origin, dimension, spacing) etc. to the predicted mask sitk volume
    # NOTE: this is 'just' meta data. The actual data is already in the matching shape since we applied the inverse transformations using tio
    sitk_vol.CopyInformation(img_vol)

    # save the prediction to the file system
    sitk.WriteImage(sitk_vol, predcac_seg_file)


def run_inference(sample: Sample, predcac_seg_file: str):
    # static parameters
    kwargs = {
        'tfx': get_tfx(cCOMMON_SPACING, cBOUNDING_BOX, cAPPLY_ZNORM),
        'patchsize': cPATCHSIZE,
        'stride': cSTRIDE,
        'threshold': cTH,
        'predcac_seg_file': predcac_seg_file
    }

    # predict sample
    predict_sample(sample, **kwargs)