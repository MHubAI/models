"""
-------------------------------------------------
MedicalHub - Run Module for Thresholding.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

from mhub.mhubio.Config import Instance, InstanceData, DataType, FileType, SEG
from mhub.mhubio.modules.runner.ModelRunner import ModelRunner

import os, numpy as np
import SimpleITK as sitk

class ThresholdingRunner(ModelRunner):
    def runModel(self, instance: Instance) -> None:

        # data
        inp_data = instance.getData(DataType(FileType.NRRD))

        # read image
        self.v(f"Reading image from {inp_data.abspath}")
        img_itk = sitk.ReadImage(inp_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)
        
        # apply threshold
        TH = self.c["TH"] if "TH" in self.c and self.c["TH"] else 300
        self.v(f"Apply th of {TH}")
        th_np = np.zeros(img_np.shape)
        th_np[img_np > TH] = 1

        # store image
        out_file = os.path.join(instance.abspath, f'thresholded.{str(TH)}.nrrd')
        self.v(f"Writing image to {out_file}")
        th_itk = sitk.GetImageFromArray(th_np)
        th_itk.CopyInformation(img_itk)
        sitk.WriteImage(th_itk, out_file)

        # meta
        meta = {
            "model": "Thresholder",
            "th": TH                    # TODO: standardize (as with the whole DataType usecase & filtering!)
        }

        # create output data
        seg_data_type = DataType(FileType.NRRD, SEG + meta)           
        seg_data = InstanceData(out_file, type=seg_data_type)
        instance.addData(seg_data)