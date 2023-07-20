"""
-----------------------------------------------------------
GC / MHub - Run Module for the GC NNUnet Pancreas Algorithm
-----------------------------------------------------------

-----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, FileType, CT, SEG, IO, Meta
import os, subprocess, shutil

from pathlib import Path

from process import PDACDetectionContainer

# TODO should move to MHubio/core/templates.py
HEATMAP     = Meta(mod="heatmap")

# @IO.Config('output_dir', str, "/app/tmp/gc_nnunet_pancreas/", the='directory to output the segmentation and the heatmap')
class GCNNUnetPancreasRunner(Module):

    # output_dir: str

    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the="input data")
    @IO.Output('heatmap', 'heatmap.mha', 'mha:mod=heatmap:model=GCNNUnetPancreas', data="in_data",
               the="heatmap of the pancreatic tumor likelihood")
    @IO.Output('segmentation', 'segmentation.mha', 'mha:mod=seg:model=GCNNUnetPancreas', data="in_data",
               the="segmentation of the pancreas, with the following classes: 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, 7-renal vein")
    # @IO.Output('vei', 'Veins.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=HEART', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the veins")
    # @IO.Output('art', 'Arteries.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=PULMONARY_ARTERY', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the arteries")
    # @IO.Output('pan', 'Pancreas.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=AORTA', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the pancreas")
    # @IO.Output('pdc', 'PDC.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=HEART', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the pancreatic duct")
    # @IO.Output('bdt', 'BileDuct.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=PULMONARY_ARTERY', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the bile duct")
    # @IO.Output('cys', 'Cysts.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=AORTA', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of cysts")
    # @IO.Output('rve', 'RenalVein.mha', 'mha:mod=seg:model=GCNNUnetPancreas:roi=AORTA', bundle='gc_nnunet_pancreas', in_signature=False,
    #            the="segmentation of the renal vein")
    def task(self, instance: Instance, in_data: InstanceData, heatmap: InstanceData, segmentation: InstanceData, **kwargs) -> None:
        algorithm = PDACDetectionContainer()
        #algorithm.ct_ip_dir         = Path("/input/images/")
        algorithm.ct_image          = in_data.abspath  # set as str not Path
        #algorithm.output_dir        = Path(self.output_dir)
        #algorithm.output_dir_tlm    = algorithm.output_dir / "pancreatic-tumor-likelihood-map"
        #algorithm.output_dir_seg    = algorithm.output_dir / "pancreas-anatomy-and-vessel-segmentation"
        algorithm.heatmap           = Path(heatmap.abspath)  # algorithm.output_dir_tlm / "heatmap.mha"
        algorithm.segmentation      = Path(segmentation.abspath)  #algorithm.output_dir_seg / "segmentation.mha"
        #algorithm.output_dir.mkdir(exist_ok=True, parents=True)
        #algorithm.output_dir_tlm.mkdir(exist_ok=True, parents=True)
        #algorithm.output_dir_seg.mkdir(exist_ok=True, parents=True)
        self.v(in_data.abspath)
        self.v(heatmap.abspath)
        self.v(segmentation.abspath)
        algorithm.process()
