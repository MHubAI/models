"""
-------------------------------------------------
MHub - Run Module for Platipy.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

from mhubio.core import Instance, InstanceData, DataType, FileType, CT, SEG
from mhubio.modules.runner.ModelRunner import ModelRunner

import os, subprocess

class PlatipyRunner(ModelRunner):
    def runModel(self, instance: Instance) -> None:

        # data
        inp_data = instance.getData(DataType(FileType.NIFTI, CT))

        # define model output folder
        out_dir = self.config.data.requestTempDir(label="pp-model-out")
        
        # build command
        bash_command  = ["platipy", "segmentation", "cardiac"]
        bash_command += ["-o", out_dir]     # /path/to/output_folder
        bash_command += [inp_data.abspath]  # /path/to/ct.nii.gz

        if "path_to_config_file" in self.c and self.c["path_to_config_file"]:
            self.v("Running the hybrid cardiac segmentation with config file at: " + str(self.c["path_to_config_file"]))
            bash_command += ["--config", str(self.c["path_to_config_file"])]
        else:
            self.v("Running the hybrid cardiac segmentation with default configuration.")

        # TODO: remove 
        self.v(">> run pp: ", " ".join(bash_command))

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
                "model": "Platipy",
                "roi": out_file[:-7]            # TODO: standardize (as with the whole DataType usecase & filtering!)
            }

            # create output data
            seg_data_type = DataType(FileType.NIFTI, SEG + meta)           
            seg_path = os.path.join(out_dir, out_file)
            seg_data = InstanceData(seg_path, type=seg_data_type)
            seg_data.dc.makeEntrypoint()
            instance.addData(seg_data)