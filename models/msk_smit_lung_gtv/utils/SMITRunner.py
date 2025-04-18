"""
-------------------------------------------------
MHub - Run Module for SMIT Lung GTV Segmentation
-------------------------------------------------

-------------------------------------------------
Author: Jue Jiang
Email:  jiangj1@mskcc.org
-------------------------------------------------
"""

import os, subprocess, shutil
from pathlib import Path
from mhubio.core import Instance, InstanceData, IO, Module
#from mhubio.modules.runner.ModelRunner import ModelRunner


CLI_CWD = str(Path(__file__).parents[1]) + '/src'


# Optional config parameter/s examples noted below
# @IO.Config('a_min', int, -500, the='Min frequency of image')
# @IO.Config('a_max', int, 500, the='Max frequency of image')

@IO.ConfigInput('in_data', 'nifti:mod=ct', the='supported datatype for the lung gtv segmentation model')

class SMITRunner(Module):
    
    @IO.Instance()
    @IO.Input('in_data', the='input ct scan')
    @IO.Output('out_data', 'model_Mhub_dicom.nii.gz', 'nifti:mod=seg:model=SMIT:roi=LUNG+NEOPLASM_MALIGNANT_PRIMARY', the='predicted segmentation of lung GTV')
   

    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:   
        wrapperName = 'bash_run_SMIT_mhub.sh'
        weightsName = os.path.join(CLI_CWD,'trained_weights','model.pt')
        
        # bash command for SMIT
        #bash_command  = f"source " + condaEnvActivateScript + " && source " + wrapperPath + " " + sessiondir + " " + sessiondir + " " + load_weight_name + " " + scan.abspath
        #"source", wrapperName, in_data.abspath, out_data.abspath, weightsName
        data_dir = os.path.dirname(in_data.abspath)
        save_folder = os.path.dirname(out_data.abspath)

        cmd1 = ["uv","run","-p","/app/.venv39","python","utils/gen_data_json.py",in_data.abspath]

        self.subprocess(cmd1, cwd=CLI_CWD,text=True)

        cmd2 = [
            "uv","run","-p","/app/.venv39",
            "python","run_segmentation.py","--roi_x","128","--roi_y","128","--roi_z","128",
            "--space_x","1.5","--space_y","1.5","--space_z","2.0","--data_dir",data_dir,"--out_channels","2",
            "--load_weight_name","trained_weights/model_Mhub.pt","--save_folder",save_folder,"--a_min=-500","--a_max=500","--use_smit","1"]

        # Display command on terminal
        self.log("Running SMIT")
        #self.log(">> ".join(bash_command))
        
        # run SMIT
        self.subprocess(cmd2, cwd = CLI_CWD, text=True)
