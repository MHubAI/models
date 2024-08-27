"""
-------------------------------------------------
MHub - Reorientation Module
-------------------------------------------------

-------------------------------------------------
Author: Jithendra
Email:  jithendra.kumar@bamfhealth.com
-------------------------------------------------
"""

from pathlib import Path
from mhubio.core import Module, Instance, InstanceDataCollection, InstanceData, DataType, FileType
from mhubio.core.IO import IO

import os

@IO.ConfigInput('in_datas', 'nifti:mod=mr', the="target data that will be reoriented")
class ReOrientationRunner(Module):
    """
    Reorient images to RAI (RIGHT, ANTERIOR, POSTERIOR)
    """

    def find_fsl(self, default_path="/usr/local/fsl/"):
        if "FSLDIR" not in os.environ:
            os.environ["FSLDIR"]="/usr/local/fsl"
        # The fsl.sh shell setup script adds the FSL binaries to the PATH
        self.fsl_path = Path(os.environ["FSLDIR"])
        os.environ["FSLOUTPUTTYPE"]="NIFTI_GZ"
        os.environ["FSLTCLSH"]=f"{self.fsl_path}/bin/fsltclsh"
        os.environ["FSLWISH"]=f"${self.fsl_path}/bin/fslwish"
        os.environ["FSL_SKIP_GLOBA"]="0"
        os.environ["FSLMULTIFILEQUIT"]="TRUE"
        assert Path(self.fsl_path/"bin/flirt").exists(), "FSL installation not found"

    @IO.Instance()
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path='[filename].nii.gz', dtype='nifti:task=reorientation', data='in_datas', bundle='reorientation', auto_increment=True, the="converted data")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, out_datas: InstanceDataCollection, **kwargs) -> None:

        # some sanity checks
        assert isinstance(in_datas, InstanceDataCollection)
        assert isinstance(out_datas, InstanceDataCollection)
        assert len(in_datas) == len(out_datas)

        # filtered collection must not be empty
        if len(in_datas) == 0:
            self.v(f"CONVERT ERROR: no data found in instance {str(instance)}.")
            return None

        # conversion step
        for i, in_data in enumerate(in_datas):
            out_data = out_datas.get(i)

            # for nrrd files use plastimatch
            self.find_fsl()
            reorient_command = [
                str(self.fsl_path / "bin" / "fslreorient2std"),
                str(in_data.abspath),
                str(out_data.abspath),
                "-s",
            ]
            self.v("reorienting....", reorient_command)
            self.subprocess(reorient_command, text=True)
