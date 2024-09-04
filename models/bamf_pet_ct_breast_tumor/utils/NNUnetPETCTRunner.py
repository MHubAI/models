"""
-------------------------------------------------
MHub - NNU-Net MultiModality Runner
       This is a base runner for pre-trained 
       nnunet models
-------------------------------------------------

-------------------------------------------------
Author: Jithendra Kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""
# TODO: support multi-i/o and batch processing on multiple instances

from typing import List, Optional
import os, subprocess, shutil
import SimpleITK as sitk, numpy as np
from mhubio.core import Module, Instance, InstanceData, DataType, FileType, IO, InstanceDataCollection

# TODO: add an optional evaluation pattern (regex) to IO.Config
nnunet_task_name_regex = r"Task[0-9]{3}_[a-zA-Z0-9_]+"

@IO.ConfigInput('in_ct_data', 'nifti:mod=ct', the="input ct data to run nnunet on")
@IO.ConfigInput('in_pt_data', 'nifti:mod=pt', the="input pt data to run nnunet on")
class NNUnetPETCTRunner(Module):

    nnunet_task: str = 'Task762_PET_CT_Breast'
    nnunet_model: str = '3d_fullres'
    input_data_type: DataType

    roi: str = 'LIVER,KIDNEY,URINARY_BLADDER,SPLEEN,LUNG,BRAIN,HEART,STOMACH,BREAST+FDG_AVID_TUMOR'

    @IO.Instance()
    @IO.Input('in_ct_data', the="input ct data to run nnunet on")
    @IO.Input('in_pt_data', the="input pt data to run nnunet on")
    @IO.Output("out_data", 'VOLUME_001.nii.gz', 'nifti:mod=seg:model=nnunet', the="output data from nnunet")
    def task(self, instance: Instance, in_ct_data: InstanceData,in_pt_data: InstanceData, out_data: InstanceData) -> None:
        
        # get the nnunet model to run
        self.v("Running nnUNet_predict.")
        self.v(f" > task:        {self.nnunet_task}")
        self.v(f" > model:       {self.nnunet_model}")
        self.v(f" > output data: {out_data.abspath}")

        # download weights if not found
        # NOTE: only for testing / debugging. For productiio always provide the weights in the Docker container.
        if not os.path.isdir(os.path.join(os.environ["WEIGHTS_FOLDER"], '')):
            print("Downloading nnUNet model weights...")
            bash_command = ["nnUNet_download_pretrained_model", self.nnunet_task]
            self.subprocess(bash_command, text=True)

        # bring input data in nnunet specific format
        # NOTE: only for nifti data as we hardcode the nnunet-formatted-filename (and extension) for now.
        # This model expects 2 input modalities for each image
        inp_dir = self.config.data.requestTempDir(label="nnunet-model-inp")
        inp_file = f'VOLUME_001_0000.nii.gz'
        shutil.copyfile(in_ct_data.abspath, os.path.join(inp_dir, inp_file))
        inp_file = f'VOLUME_001_0001.nii.gz'
        shutil.copyfile(in_pt_data.abspath, os.path.join(inp_dir, inp_file))

        # define output folder (temp dir) and also override environment variable for nnunet
        out_dir = self.config.data.requestTempDir(label="nnunet-model-out")
        os.environ['RESULTS_FOLDER'] = out_dir

        # symlink nnunet input folder to the input data with python
        # create symlink in python
        # NOTE: this is a workaround for the nnunet bash script that expects the input data to be in a specific folder
        #       structure. This is not the case for the mhub data structure. So we create a symlink to the input data
        #       in the nnunet input folder structure.
        os.symlink(os.environ['WEIGHTS_FOLDER'], os.path.join(out_dir, 'nnUNet'))
        
        # construct nnunet inference command
        bash_command  = ["nnUNet_predict"]
        bash_command += ["--input_folder", str(inp_dir)]
        bash_command += ["--output_folder", str(out_dir)]
        bash_command += ["--task_name", self.nnunet_task]
        bash_command += ["--model", self.nnunet_model]
        
        self.v(f" > command 1:  {bash_command}")
        # run command
        self.subprocess(bash_command, text=True)

        # output meta
        meta = {
            "model": "nnunet",
            "nnunet_task": self.nnunet_task,
            "nnunet_model": self.nnunet_model,
            "roi": self.roi
        }

        # get output data
        out_file = f'VOLUME_001.nii.gz'
        out_path = os.path.join(out_dir, out_file)

        # copy output data to instance
        shutil.copyfile(out_path, out_data.abspath)

        # update meta dynamically
        out_data.type.meta += meta
