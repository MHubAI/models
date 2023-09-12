"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Spider baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
from mhubio.core import Instance, InstanceData, IO, Module

from pathlib import Path
import shutil
import json
import os

# The GC Spider baseline algorithm is imported as a self-contained algorithm class with an execute method
from TestSpine import Module as SpiderAlgorithm


class SpiderBaselineRunner(Module):

    SPIDER_DATA_DIR = Path(os.environ["VERSEG_BASEDIR"])
    SPIDER_INTERNAL_DATASET_PATH = SPIDER_DATA_DIR / "datasets" / "spider_input"
    SPIDER_EXPERIMENT_DIR = SPIDER_DATA_DIR / "experiments" / "SPIDER-Baseline"
    SPIDER_INTERNAL_OUTPUT_DIR = SPIDER_EXPERIMENT_DIR / "results" / "spider_input"

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=mr', the='input sagittal spine MRI')
    @IO.Output('out_data', 'spider_baseline_vertebrae_segmentation.mha', 'mha:mod=seg:model=SpiderBaselineSegmentation', 'in_data', the='Spider baseline vertebrae segmentation for the input sagittal spine MRI')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        in_data_path = Path(in_data.abspath)
        internal_img_path = self.SPIDER_INTERNAL_DATASET_PATH / "images" / "input_img.mha"
        self.v(f"Copying data files to internal SPIDER data structure: {in_data_path} -> {internal_img_path}")
        shutil.copy(str(in_data_path), str(internal_img_path))

        self.v("Run the SPIDER-Baseline algorithm")
        # The algorithm configuration is configured to run on an internal data folder structure and is defined
        # further configured by the JSON files in the Dockerfile
        # The algorithm parameters specify the following:
        # * The first argument specifies the folder to look for the model files
        # * The epoch specifies the model weights file to use, which should be something like: `999999.pt`
        # * The dataset is set to the internal SPIDER data structure
        # * The surface erosion threshold is set to the default for the MR modality (as it is always assumed to have MR modality)
        # * The original image dimensions are used to generate the output segmentation
        SpiderAlgorithm(
            [
                "SPIDER-Baseline",
                "--epoch", "999999",
                "--dataset", "spider_input",
                "--surface_erosion_threshold", "-2000",
                "--export_original"
            ]
        ).execute()

        source_output_file = self.SPIDER_INTERNAL_OUTPUT_DIR / "input_img_total_segmentation_original.mha"
        self.v(f"Move generated segmentation output from: {source_output_file} -> {out_data.abspath}")
        shutil.move(str(source_output_file), out_data.abspath)

        # Run internal output cleanup
        shutil.rmtree(str(self.SPIDER_INTERNAL_OUTPUT_DIR))
