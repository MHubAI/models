"""
-------------------------------------------------
MHub - NNU-Net Runner v2
       Custom Runner for pre-trained nnunet v2 models.
-------------------------------------------------

-------------------------------------------------
Author: Jithendra kumar
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

import os, shutil
from mhubio.core import Module, Instance, InstanceData, DataType, FileType, IO


@IO.ConfigInput('in_data', 'nifti', the="input data to run nnunet on")
@IO.Config('nnunet_dataset', str, None, the='nnunet dataset name')
@IO.Config('nnunet_config', str, None, the='nnunet model name (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)')
@IO.Config('roi', str, None, the='roi or comma separated list of roi the nnunet segments')
class NNUnetRunnerV2(Module):

    nnunet_dataset: str
    nnunet_config: str
    input_data_type: DataType
    roi: str

    @IO.Instance()
    @IO.Input("in_data", the="input data to run nnunet on")
    @IO.Output("out_data", 'VOLUME_001.nii.gz', 'nifti:mod=seg:model=nnunet', data='in_data', the="output data from nnunet")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        
        # get the nnunet model to run
        self.v("Running nnUNetv2_predict.")
        self.v(f" > dataset:     {self.nnunet_dataset}")
        self.v(f" > config:      {self.nnunet_config}")
        self.v(f" > input data:  {in_data.abspath}")
        self.v(f" > output data: {out_data.abspath}")

        # download weights if not found
        # NOTE: only for testing / debugging. For productiio always provide the weights in the Docker container.
        if not os.path.isdir(os.path.join(os.environ["WEIGHTS_FOLDER"], '')):
            print("Downloading nnUNet model weights...")
            bash_command = ["nnUNet_download_pretrained_model", self.nnunet_dataset]
            self.subprocess(bash_command, text=True)

        # bring input data in nnunet specific format
        # NOTE: only for nifti data as we hardcode the nnunet-formatted-filename (and extension) for now.
        assert in_data.type.ftype == FileType.NIFTI
        assert in_data.abspath.endswith('.nii.gz')
        inp_dir = self.config.data.requestTempDir(label="nnunet-model-inp")
        inp_file = f'VOLUME_001_0000.nii.gz'
        shutil.copyfile(in_data.abspath, os.path.join(inp_dir, inp_file))

        # define output folder (temp dir) and also override environment variable for nnunet
        out_dir = self.config.data.requestTempDir(label="nnunet-model-out")
        os.environ['nnUNet_results'] = out_dir

        # symlink nnunet input folder to the input data with python
        # create symlink in python
        # NOTE: this is a workaround for the nnunet bash script that expects the input data to be in a specific folder
        #       structure. This is not the case for the mhub data structure. So we create a symlink to the input data
        #       in the nnunet input folder structure.
        # os.symlink(os.environ['WEIGHTS_FOLDER'], os.path.join(out_dir, 'nnUNet'))
        os.symlink(os.path.join(os.environ['WEIGHTS_FOLDER'], self.nnunet_dataset), os.path.join(out_dir, self.nnunet_dataset))
        
        # NOTE: instead of running from commandline this could also be done in a pythonic way:
        #       `nnUNet/nnunet/inference/predict.py` - but it would require
        #       to set manually all the arguments that the user is not intended
        #       to fiddle with; so stick with the bash executable

        # construct nnunet inference command
        bash_command  = ["nnUNetv2_predict"]
        bash_command += ["-i", str(inp_dir)]
        bash_command += ["-o", str(out_dir)]
        bash_command += ["-d", self.nnunet_dataset]
        bash_command += ["-c", self.nnunet_config]

        self.v(f" > bash_command:     {bash_command}")
        # run command
        self.subprocess(bash_command, text=True)

        # output meta
        meta = {
            "model": "nnunet",
            "nnunet_dataset": self.nnunet_dataset,
            "nnunet_config": self.nnunet_config,
            "roi": self.roi
        }

        # get output data
        out_file = f'VOLUME_001.nii.gz'
        out_path = os.path.join(out_dir, out_file)

        # copy output data to instance
        shutil.copyfile(out_path, out_data.abspath)

        # update meta dynamically
        out_data.type.meta += meta
