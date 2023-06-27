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

import pipeline.tils_pipeline as tils_pipeline
import algorithm.rw as rw


class TigerLB2Runner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'tiff', the='input whole slide image Tiff')
    @IO.Output('out_data', 'tiger_lb2_tils_score.json', 'json:model=TigerLB2TilsScore', 'in_data', the='TIGER LB2 Tils score')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        wsi_filepath = Path(in_data.abspath)
        wsi_mri = rw.open_multiresolutionimage_image(wsi_filepath)

        print(f"Input WSI: {wsi_filepath}")

        tils_score_writer = rw.TilsScoreWriter(Path(out_data.abspath))
        tils_score = tils_pipeline.run_tils_pipeline(wsi_mri)

        # write tils score
        self.v(f"Writing tils score to {out_data.abspath}")
        tils_score_writer.set_tils_score(tils_score=tils_score)
        tils_score_writer.save()
