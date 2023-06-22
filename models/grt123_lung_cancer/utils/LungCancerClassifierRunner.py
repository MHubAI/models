"""
----------------------------------------------------------
Mhub / DIAG - Run Module for grt123 Lung Cancer Classifier
----------------------------------------------------------

----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
----------------------------------------------------------
"""
import torch.cuda
from mhubio.core import Instance, InstanceData, IO, Module

from pathlib import Path
import numpy as np
import SimpleITK as sitk

import torch

import main


@IO.Config('n_preprocessing_workers', int, 6, the="number of preprocessing workers to use for the grt123 lung mask preprocessor")
@IO.Config('tmp_path', str, "/app/tmp", the="the path to write intermediate grt123 files to")
class LungCancerClassifierRunner(Module):

    n_preprocessing_workers: int
    tmp_path: str

    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the='input ct scan')
    @IO.Output('out_data', 'grt123_lung_cancer_findings.json', 'json:model=grt123LungCancerClassification', 'in_data', the='predicted nodules and lung cancer findings of the lung lobe')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:

        tmp_path = Path(self.tmp_path)
        tmp_output_bbox_dir = tmp_path / "bbox"
        tmp_output_prep_dir = tmp_path / "prep"
        tmp_output_bbox_dir.mkdir(exist_ok=True, parents=True)
        tmp_output_prep_dir.mkdir(exist_ok=True, parents=True)

        n_gpu = 1 if torch.cuda.is_available() else 0

        # apply grt123 algorithm
        results = main.main(
            skip_detect=False,
            skip_preprocessing=False,
            datapath=str(Path(in_data.abspath).parent),
            outputdir=str(tmp_path),
            output_bbox_dir=str(tmp_output_bbox_dir),
            output_prep_dir=str(tmp_output_prep_dir),
            n_gpu=n_gpu,
            n_worker_preprocessing=self.n_preprocessing_workers,
            data_filter=r".*.mha"
        )

        # store classification results
        self.v(f"Writing classification results to {out_data.abspath}")
        assert len(results) > 0, "LungCancerClassifierRunner - Always expects at least one output report"
        results[0].to_file(out_data.abspath)
