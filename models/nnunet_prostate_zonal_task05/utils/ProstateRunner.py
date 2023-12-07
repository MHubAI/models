import os, shutil
from mhubio.core import Module, Instance, InstanceData, IO

@IO.Config('use_tta', bool, False, the='flag to enable test time augmentation')
@IO.Config('nnunet_model', str, None, the='nnunet model name (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)')
class ProstateRunner(Module):

    use_tta: bool
    nnunet_model: str

    @IO.Instance()
    @IO.Input('T2', 'nifti:part=T2', the="T2 image")
    @IO.Input('ADC', 'nifti:part=ADC:resampled_to=T2', the="ADC image resampled to T2")
    @IO.Output('P', 'VOLUME_001.nii.gz', 'nifti:mod=seg:model=nnunet_t005_prostate:roi=PROSTATE_PERIPHERAL_ZONE,PROSTATE_TRANSITION_ZONE', bundle='nnunet-out', the="Prostate segmentation")
    def task(self, instance: Instance, T2: InstanceData, ADC: InstanceData, P: InstanceData) -> None:

        # copy input files to align with the nnunet input folder and file name format
        # T2:  0000 
        # ADC: 0001
        inp_dir = self.config.data.requestTempDir(label="nnunet-model-inp")
        inp_file_T2 =  f'VOLUME_001_0000.nii.gz'
        inp_file_ADC = f'VOLUME_001_0001.nii.gz'
        shutil.copyfile(T2.abspath, os.path.join(inp_dir, inp_file_T2))
        shutil.copyfile(ADC.abspath, os.path.join(inp_dir, inp_file_ADC))

        # define output folder (temp dir) and also override environment variable for nnunet
        assert P.bundle is not None, f"Output bundle is required: {str(P)}"
        os.environ['RESULTS_FOLDER'] = P.bundle.abspath

        # symlink nnunet input folder to the input data with python
        # create symlink in python
        # NOTE: this is a workaround for the nnunet bash script that expects the input data to be in a specific folder
        #       structure. This is not the case for the mhub data structure. So we create a symlink to the input data
        #       in the nnunet input folder structure.
        os.symlink(os.environ['WEIGHTS_FOLDER'], os.path.join(P.bundle.abspath, 'nnUNet'))

        # construct nnunet inference command
        bash_command  = ["nnUNet_predict"]
        bash_command += ["--input_folder", str(inp_dir)]
        bash_command += ["--output_folder", str(P.bundle.abspath)]
        bash_command += ["--task_name", 'Task005_Prostate']
        bash_command += ["--model", self.nnunet_model]
        
        # optional / customizable arguments
        if not self.use_tta:
            bash_command += ["--disable_tta"]

        # run command
        self.subprocess(bash_command, text=True)