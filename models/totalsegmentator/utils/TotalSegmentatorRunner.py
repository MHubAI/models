"""
-------------------------------------------------
MHub - Run Module for TotalSegmentator.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

from mhubio.modules.runner.ModelRunner import ModelRunner
from mhubio.Config import Instance, InstanceData, DataType, FileType, SEG

import os, subprocess

class TotalSegmentatorRunner(ModelRunner):
    def runModel(self, instance: Instance) -> None:

        # configuration
        use_fast_mode = "use_fast_mode" in self.c and self.c["use_fast_mode"]
        use_multi_label_output = "use_multi_label_output" in self.c and self.c["use_multi_label_output"]
        
        # data
        inp_data = instance.getData(DataType(FileType.NIFTI))

        # define model output folder
        out_dir = self.config.data.requestTempDir(label="ts-model-out")
        
        # build command
        bash_command  = ["TotalSegmentator"]
        bash_command += ["-i", inp_data.abspath]
    
        # multi-label output (one nifti file containing all labels instead of one nifti file per label)
        if use_multi_label_output:
            self.v("Generating multi-label output ('--ml')")
            bash_command += ["-o", os.path.join(out_dir, 'segmentations.nii.gz')]
            bash_command += ["--ml"]
        else:
            self.v("Generating single-label output (default)")
            bash_command += ["-o", out_dir]

        # fast mode
        if use_multi_label_output:
            self.v("Running TotalSegmentator in fast mode ('--fast', 3mm)")
            bash_command += ["--fast"]
        else:
            self.v("Running TotalSegmentator in default mode (1.5mm)")

        # TODO: remove 
        self.v(">> run: ", " ".join(bash_command))

        # run the model
        bash_return = subprocess.run(bash_command, check=True, text=True)

        # add output data
        for out_file in os.listdir(out_dir):

            # ignore non nifti files
            if out_file[-7:] != ".nii.gz":
                self.v(f"IGNORE OUTPUT FILE {out_file}")
                continue

            # meta
            meta = {
                "model": "TotalSegmentator",
                "roi": out_file[:-7]            # TODO: standardize (as with the whole DataType usecase & filtering!)
            }

            # create output data
            seg_data_type = DataType(FileType.NIFTI, SEG + meta)           
            seg_path = os.path.join(out_dir, out_file)
            seg_data = InstanceData(seg_path, type=seg_data_type)
            seg_data.base = "" # required since path is external (will be fixed soon)
            instance.addData(seg_data)  