"""
------------------------------------------------------
Mhub / DIAG - Run Module for Xie2020 Lobe Segmentation
------------------------------------------------------

------------------------------------------------------
Author: Sil van de Leemput, Leonard Nürnberg
Email:  s.vandeleemput@radboudumc.nl
        leonard.nuernberg@maastrichtuniversity.nl
------------------------------------------------------
"""

from mhubio.core import Instance, InstanceData, IO, Module
from pathlib import Path

CLI_PATH = Path(__file__).parent / "run38.py"

@IO.ConfigInput('in_data', 'nifti|nrrd|mha:mod=ct', the='supported datatypes for the lobes segmentation model')
class LobeSegmentationRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', the='input ct scan')
    @IO.Output('out_data', 'xie2020lobeseg.mha', 'mha:mod=seg:model=Xie2020LobeSegmentation:roi=LEFT_UPPER_LUNG_LOBE,LEFT_LOWER_LUNG_LOBE,RIGHT_UPPER_LUNG_LOBE,RIGHT_LOWER_LUNG_LOBE,RIGHT_MIDDLE_LUNG_LOBE', bundle='model', the='predicted segmentation of the lung lobes')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        # NOTE input data originally was specified for MHA/MHD and could be extended for DICOM

        # Call the Xie2020 Lobe Segmentation in a separate python3.8 venv
        cmd = [
            "uv", "run", "-p", ".venv38", "python",
            str(CLI_PATH),
            in_data.abspath,
            out_data.abspath
        ]
        
        self.subprocess(cmd, text=True)
