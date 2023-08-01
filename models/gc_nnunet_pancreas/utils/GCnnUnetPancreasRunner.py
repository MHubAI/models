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

from process import PDACDetectionContainer

# TODO should move to MHubio/core/templates.py
HEATMAP     = Meta(mod="heatmap")

class GCNNUnetPancreasRunner(Module):
    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the="input data")
    @IO.Output('heatmap', 'heatmap.mha', 'mha:mod=heatmap:model=GCNNUnetPancreas', data="in_data",
               the="heatmap of the pancreatic tumor likelihood")
    @IO.Output('segmentation', 'segmentation.mha', 'mha:mod=seg:model=GCNNUnetPancreas', data="in_data",
               the="segmentation of the pancreas, with the following classes: "
                   "1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, 7-renal vein")
    def task(self, instance: Instance, in_data: InstanceData, heatmap: InstanceData, segmentation: InstanceData, **kwargs) -> None:
        algorithm = PDACDetectionContainer()
        algorithm.ct_image          = in_data.abspath  # set as str not Path
        algorithm.heatmap           = Path(heatmap.abspath)
        algorithm.segmentation      = Path(segmentation.abspath)
        algorithm.process()
