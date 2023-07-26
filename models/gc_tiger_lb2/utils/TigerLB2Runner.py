"""
------------------------------------------------
Mhub / DIAG - Run Module for Tiger LB2 Algorithm
------------------------------------------------

------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

import subprocess as sp
import sys


class TigerLB2Runner(Module):

    CLI_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "tiger_lb2_cli.py"

    @IO.Instance()
    @IO.Input('in_data', 'tiff:mod=sm', the='input whole slide image Tiff')
    @IO.Output('out_data', 'gc_tiger_lb2_tils_score.json', 'json:model=TigerLB2TilsScore', 'in_data', the='TIGER LB2 Tils score')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        assert torch.cuda.is_available(), "Error: TigerLB2Runner requires CUDA to be available!"

        proc = sp.run(
            [
                sys.executable,
                str(self.CLI_SCRIPT_PATH),
                in_data.abspath,
                out_data.abspath,
            ]
        )

        assert proc.returncode == 0, f"Something went wrong when calling {self.CLI_SCRIPT_PATH}, got return code: {proc.returncode}"
