"""
-----------------------------------------------------------------------
Mhub / DIAG - Run Module for AutoPET false positive reduction algorithm
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""
import sys
from pathlib import Path

from mhubio.core import Instance, InstanceData, IO, Module


CLI_PATH = Path(__file__).parent.parent / "src" / "cli.py"


class AutoPETRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_pet', 'mha|nifti:mod=pt', the='input FDG PET scan')
    @IO.Input('in_data_ct', 'mha|nifti:mod=ct:resampled=true', the='input FDG CT scan, resampled to FDG PET scan')
    @IO.Output('out_data', 'tumor_segmentation.mha', 'mha:mod=seg:model=AutoPET:roi=NEOPLASM_MALIGNANT_PRIMARY', bundle='model', the='predicted tumor segmentation within the input FDG PET/CT scan')
    def task(self, instance: Instance, in_data_pet: InstanceData, in_data_ct: InstanceData, out_data: InstanceData) -> None:
        # Call the AutoPET FPR CLI
        # A CLI was used here to ensure the mhub framework properly captures the nnUNet stdout output
        cmd = [
            sys.executable,
            str(CLI_PATH),
            in_data_pet.abspath,
            in_data_ct.abspath,
            out_data.abspath
        ]
        self.subprocess(cmd, text=True)
