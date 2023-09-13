"""
------------------------------------------------
Mhub / DIAG - Run Module for Tiger LB2 Algorithm
------------------------------------------------

------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, Meta

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch

import subprocess as sp
import sys
import json


@ValueOutput.Name('til_score')
@ValueOutput.Meta(Meta(key="value"))
@ValueOutput.Label('TIL score')
@ValueOutput.Type(float)
@ValueOutput.Description('percentage of stromal area covered by tumour infiltrating lymphocytes. Values between 0 (percent) to 100 (percent).')
class TilScoreOutput(ValueOutput):
    pass


class TigerLB2Runner(Module):

    CLI_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "tiger_lb2_cli.py"

    @IO.Instance()
    @IO.Input('in_data', 'tiff:mod=sm', the='input whole slide image Tiff')
    @IO.Output('out_data', 'gc_tiger_lb2_til_score.json', 'json:model=TigerLB2TILScore', 'in_data', the='TIGER LB2 TIL score')
    @IO.OutputData('til_score', TilScoreOutput, data='in_data', the='TIGER LB2 TIL score - percentage of stromal area covered by tumour infiltrating lymphocytes. Values between 0-100 (percent).')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData, til_score: TilScoreOutput) -> None:
        assert torch.cuda.is_available(), "Error: TigerLB2Runner requires CUDA to be available!"

        # Execute the Tiger LB2 Algorithm through a Python subprocess
        proc = sp.run(
            [
                sys.executable,
                str(self.CLI_SCRIPT_PATH),
                in_data.abspath,
                out_data.abspath,
            ]
        )

        assert proc.returncode == 0, f"Something went wrong when calling {self.CLI_SCRIPT_PATH}, got return code: {proc.returncode}"
        out_data.confirm()

        # export output til score as data as well
        with open(out_data.abspath, "r") as f:
            til_score.value = json.load(f)
        assert isinstanceof(til_score.value, float)
