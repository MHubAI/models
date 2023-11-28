"""
-------------------------------------------------
MHub - run the LungMask pipeline
-------------------------------------------------

-------------------------------------------------
Author: Dennis Bontempi
Email:  d.bontempi@maastrichtuniversity.nl
-------------------------------------------------
"""

import os, subprocess, shutil
from mhubio.core import Instance, InstanceData, IO
from mhubio.modules.runner.ModelRunner import ModelRunner

@IO.Config('batchsize', int, 64, the='Number of slices to be processed simultaneously. A smaller batch size requires less memory but may be slower.')

class LungMaskRunner(ModelRunner):
    
    batchsize: int

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Output('roi1_lungs', 'roi1_lungs.nii.gz', 'nifti:mod=seg:model=LungMask:roi=RIGHT_LUNG,LEFT_LUNG', bundle='model', the='predicted segmentation of the lungs')
    @IO.Output('roi2_lunglobes', 'roi2_lunglobes.nii.gz', 'nifti:mod=seg:model=LungMask:roi=LEFT_UPPER_LUNG_LOBE,LEFT_LOWER_LUNG_LOBE,RIGHT_UPPER_LUNG_LOBE,RIGHT_MIDDLE_LUNG_LOBE,RIGHT_LOWER_LUNG_LOBE', bundle='model', the='predicted segmentation of the lung lobes')
    def task(self, instance: Instance, image: InstanceData, roi1_lungs: InstanceData, roi2_lunglobes: InstanceData) -> None:
        
        # bash command for the lung segmentation *
        bash_command  = ["lungmask"]
        bash_command += [image.abspath]         # path to the input_file
        bash_command += [roi1_lungs.abspath]    # path to the output file
        bash_command += ["--modelname", "R231"] # specify lung seg model

        self.v("Running the lung segmentation.")
        self.v(">> run lungmask (R231): ", " ".join(bash_command))
        
        # run the lung segmentation model
        self.subprocess(bash_command, text=True)


        # bash command for the lung lobes segmentation (fusion)
        bash_command  = ["lungmask"]
        bash_command += [image.abspath]                   # path to the input_file
        bash_command += [roi2_lunglobes.abspath]          # path to the output file
        bash_command += ["--modelname", "LTRCLobes_R231"] # specify lung lobes seg model

        self.v("Running the lung lobes segmentation (with model fusion).")
        self.v(">> run lungmask (LTRCLobes_R231): ", " ".join(bash_command))
        
        # run the lung lobes segmentation model
        self.subprocess(bash_command, text=True)