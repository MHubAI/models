"""
-------------------------------------------------
MHub - Run Module for SMIT
-------------------------------------------------

-------------------------------------------------
Author: Jue Jiang
Email:  jiangj1@mskcc.org
-------------------------------------------------
"""

import os, subprocess, shutil
from mhubio.core import Instance, InstanceData, IO
from mhubio.modules.runner.ModelRunner import ModelRunner

# Optional config parameter/s examples noted below
# @IO.Config('a_min', int, -500, the='Min frequency of image')
# @IO.Config('a_max', int, 500, the='Max frequency of image')

class SMITRunner(ModelRunner):
    
    # a_min: int
    # a_max: int

    @IO.Instance()
    @IO.Input('scan', 'nifti:mod=ct',  the='input ct scan')
    @IO.Output('gtv_mask', 'gtv_mask.nii.gz', 'nifti:mod=seg:model=SMIT:roi=LUNG+NEOPLASM_MALIGNANT_PRIMARY',data='scan', the='predicted lung GTV')
    def task(self, instance: Instance, scan: InstanceData, gtv_mask: InstanceData) -> None:
        
        workDir = os.path.join(os.environ['WORK_DIR'],'models','msk_smit_lung_gtv','src')   # Needs to be defined in docker file as ENV WORK_DIR=path_to_dir e.g. /app/models/SMIT/workDir
        #wrapperInstallDir = os.path.join(workDir,'CT_Lung_SMIT')
        #condaEnvDir = os.path.join(wrapperInstallDir,'conda-pack')
        #condaEnvActivateScript = os.path.join(condaEnvDir, 'bin', 'activate')
        wrapperPath = os.path.join(workDir,'bash_run_SMIT_Segmentation.sh')
        load_weight_name = os.path.join(workDir,'trained_weights','model.pt')

        sessionPath = os.path.join(workDir, 'session')
        os.makedirs(sessionPath, exist_ok = True)

        subj = os.path.basename(scan.abspath)   # Was originally dcmdir so might want to change
        sessiondir = os.path.join(sessionPath,subj)
        os.makedirs(sessiondir,exist_ok=True)
        
        # bash command for SMIT
        bash_command  = f"source " + condaEnvActivateScript + " && source " + wrapperPath + " " + sessiondir + " " + sessiondir + " " + load_weight_name + " " + scan.abspath

        # Display command on terminal
        self.log("Running SMIT")
        self.log(">> ".join(bash_command))
        
        # run SMIT
        self.subprocess(bash_command, text=True)
