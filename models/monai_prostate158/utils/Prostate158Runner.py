"""
-------------------------------------------------
MHub - MONAI Prostate158 Runner
-------------------------------------------------

-------------------------------------------------
Author: Cosmin Ciausu
Email:  cciausu97@gmail.com
-------------------------------------------------
"""
# TODO: support multi-i/o and batch processing on multiple instances

from typing import List, Optional
import os, subprocess, shutil, glob, sys
import SimpleITK as sitk, numpy as np
from mhubio.core import Module, Instance, InstanceData, DataType, FileType, IO
from mhubio.modules.runner.ModelRunner import ModelRunner
import json

@IO.Config('apply_center_crop', bool, True, the='flag to apply center cropping to input_image')
class Prostate158Runner(Module):

    apply_center_crop : bool

    @IO.Instance()
    @IO.Input("in_data", 'nifti:mod=mr', the="input T2 sequence data to run prostate158 on")
    @IO.Output('out_data', 'monai_prostate158.nii.gz', 
               'nifti:mod=seg:model=MonaiProstate158:roi=PROSTATE_TRANSITION_ZONE,PROSTATE_PERIPHERAL_ZONE', 
               data='in_data', bundle='model', the="predicted segmentation model")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:

        # bring input data in nnunet specific format
        # NOTE: only for nifti data as we hardcode the nnunet-formatted-filename (and extension) for now.
        assert in_data.type.ftype == FileType.NIFTI
        assert in_data.abspath.endswith('.nii.gz')
        datalist = [in_data.abspath]

        if self.apply_center_crop:
            in_dir_cropped = self.config.data.requestTempDir(label="monai-crop-in")
            in_data_processed = os.path.join(in_dir_cropped, "image_cropped.nii.gz")
            self.subprocess([sys.executable, f"{os.path.join(os.environ['BUNDLE_ROOT'], 'scripts', 'center_crop.py')}",
            "--file_name", in_data.abspath, "--out_name",in_data_processed], text=True)
            datalist = [in_data_processed]

        # define output folder (temp dir) and also override environment variable for nnunet
        out_dir = self.config.data.requestTempDir(label="monai-model-out")
       
        bash_command = [sys.executable, 
        "-m", "monai.bundle", "run", "evaluating"]
        bash_command += ["--meta_file", os.path.join(os.environ['BUNDLE_ROOT'], "configs", "metadata.json")]
        bash_command += ["--config_file", os.path.join(os.environ['BUNDLE_ROOT'], "configs", "inference.json")]
        bash_command += ["--datalist", str(datalist)]
        bash_command += ["--output_dir", out_dir]
        bash_command += ["--bundle_root", os.environ['BUNDLE_ROOT']]
        bash_command += ["--dataloader#num_workers", "0"]
        print(bash_command)
        self.subprocess(bash_command, text=True)

        # get output data
        out_path = glob.glob(os.path.join(out_dir, "**", 
        "*.nii.gz"), recursive=True)[0]

        if self.apply_center_crop:
            out_dir_padded = self.config.data.requestTempDir(label="monai-padded-out")
            out_data_padded = os.path.join(out_dir_padded, "seg_padded.nii.gz")
            paddedFilter = sitk.ConstantPadImageFilter()
            seg_image = sitk.ReadImage(out_path)
            t2_image = sitk.ReadImage(in_data.abspath)
            out_seg_shape = sitk.GetArrayFromImage(seg_image).shape
            t2_image_shape = sitk.GetArrayFromImage(t2_image).shape
            x_bound_lower =  int((t2_image_shape[2] - out_seg_shape[2])/2)
            x_bound_upper =  int(int(t2_image_shape[2] - out_seg_shape[2])/2 + ((t2_image_shape[2] - out_seg_shape[2]) % 2))
            y_bound_lower = int((t2_image_shape[1] - out_seg_shape[1])/2)
            y_bound_upper = int(int(t2_image_shape[1] - out_seg_shape[1])/2 + ((t2_image_shape[1] - out_seg_shape[1]) % 2))
            paddedFilter.SetConstant(0)
            paddedFilter.SetPadLowerBound([x_bound_lower, y_bound_lower, 0])
            paddedFilter.SetPadUpperBound([x_bound_upper, y_bound_upper, 0])
            padded_img = paddedFilter.Execute(seg_image)
            sitk.WriteImage(padded_img, out_data_padded)
            out_path = out_data_padded

        # copy output data to instance
        shutil.copyfile(out_path, out_data.abspath)
