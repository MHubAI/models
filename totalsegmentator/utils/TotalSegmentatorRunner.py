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
        
        # data
        inp_data = instance.getData(DataType(FileType.NIFTI))

        # define model output folder
        out_dir = self.config.data.requestTempDir(label="ts-model-out")
        
        # build command
        bash_command  = ["TotalSegmentator"]
        bash_command += ["-i", inp_data.abspath]
        bash_command += ["-o", out_dir]

        #platipy segmentation cardiac -o /app/tmp/a707f22b-79b0-4c89-95ae-aafb6b6adda1 -i /app/data/sorted/1.3.6.1.4.1.14519.5.2.1.7009.9004.131908895673988322984492867976/image.nii.gz

        if self.c["use_fast_mode"]:
            self.v("Running TotalSegmentator in fast mode ('--fast', 3mm): ")
            bash_command += ["--fast"]
        else:
            self.v("Running TotalSegmentator in default mode (1.5mm)")

        # TODO: remove 
        self.v(">> run ts: ", " ".join(bash_command))

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