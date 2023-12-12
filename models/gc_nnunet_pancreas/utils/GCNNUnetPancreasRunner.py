"""
-----------------------------------------------------------
GC / MHub - Run Module for the GC NNUnet Pancreas Algorithm
-----------------------------------------------------------

-----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, Meta, IO

from pathlib import Path
import SimpleITK
import numpy as np
import sys


CLI_PATH = Path(__file__).parent / "cli.py"


class GCNNUnetPancreasRunner(Module):
    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the="input data")
    @IO.Output('heatmap', 'heatmap.mha', 'mha:mod=heatmap:model=GCNNUnetPancreas', data="in_data",
               the="heatmap of the pancreatic tumor likelihood")
    @IO.Output('segmentation', 'segmentation.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=VEIN,ARTERY,PANCREAS,PANCREATIC_DUCT,BILE_DUCT,PANCREAS+CYST,RENAL_VEIN', data="in_data",
               the="original segmentation of the pancreas, with the following classes: "
                   "0-background, 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, 7-renal vein")
    def task(self, instance: Instance, in_data: InstanceData, heatmap: InstanceData, segmentation: InstanceData, **kwargs) -> None:
        # Call the PDAC CLI
        cmd = [
            sys.executable,
            str(CLI_PATH),
            in_data.abspath,
            heatmap.abspath,
            segmentation.abspath
        ]
        self.subprocess(cmd, text=True)
