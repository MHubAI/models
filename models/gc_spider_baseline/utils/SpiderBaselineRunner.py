"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Spider baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
from pathlib import Path
from typing import Dict
import shutil
import os
import json

import SimpleITK
import numpy as np

from mhubio.core import Instance, InstanceData, IO, Module, CT, MR, Meta

# The GC Spider baseline algorithm is imported as a self-contained algorithm class with an execute method
from TestSpine import Module as SpiderAlgorithm


SPIDER_DATA_DIR = Path(os.environ["VERSEG_BASEDIR"])
SPIDER_INTERNAL_DATASET_PATH = SPIDER_DATA_DIR / "datasets" / "spider_input"
SPIDER_EXPERIMENT_DIR = SPIDER_DATA_DIR / "experiments" / "SPIDER-Baseline"
SPIDER_INTERNAL_OUTPUT_DIR = SPIDER_EXPERIMENT_DIR / "results" / "spider_input"
SPIDER_EXPERIMENT_ARGUMENTS_FILE = SPIDER_EXPERIMENT_DIR / "arguments.json"
SPIDER_INTERNAL_DATASET_META_FILE = SPIDER_INTERNAL_DATASET_PATH / "metadata.json"


@IO.Config('traversal_direction_up', bool, True, the="Direction to traverse the image")
@IO.ConfigInput('in_data', 'mha:mod=ct|mr', the='supported datatypes for the spider baseline model')
class SpiderBaselineRunner(Module):

    traversal_direction_up: bool

    @IO.Instance()
    @IO.Input('in_data', the='input sagittal spine image (MR or CT)')
    @IO.Output(
        'out_data_raw',
        'spider_baseline_vertebrae_segmentation_raw.mha',
        'mha:mod=seg:model=SpiderBaselineSegmentation:seg=raw',
        data='in_data',
        bundle='model',
        the='Spider baseline vertebrae segmentation for the input sagittal spine MRI or CT. Raw segmentation output: '
            '0: background '
            '1-24: different vertebrae numbered from the bottom (i.e. L5 = 1, C1 = 24) '
            '100: spinal canal '
            '101-124 for partially visible vertebrae '
            '201-224: different intervertebral discs (also numbered from the bottom, i.e. L5/S1 = 201, C2/C3 = 224) '
    )
    @IO.Output(
        'out_data',
        'spider_baseline_vertebrae_segmentation.mha',
        'mha:mod=seg:model=SpiderBaselineSegmentation:seg=remapped:roi=VERTEBRAE_L5,VERTEBRAE_L4,VERTEBRAE_L3,VERTEBRAE_L2,'
        'VERTEBRAE_L1,VERTEBRAE_T12,VERTEBRAE_T11,VERTEBRAE_T10,VERTEBRAE_T9,VERTEBRAE_T8,VERTEBRAE_T7,VERTEBRAE_T6,VERTEBRAE_T5,'
        'VERTEBRAE_T4,VERTEBRAE_T3,VERTEBRAE_T2,VERTEBRAE_T1,VERTEBRAE_C7,VERTEBRAE_C6,VERTEBRAE_C5,VERTEBRAE_C4,VERTEBRAE_C3,VERTEBRAE_C2,VERTEBRAE_C1,'
        'VERTEBRAE_DISK_L5S1,VERTEBRAE_DISK_L4L5,VERTEBRAE_DISK_L3L4,VERTEBRAE_DISK_L2L3,VERTEBRAE_DISK_L1L2,VERTEBRAE_DISK_T12L1,VERTEBRAE_DISK_T11T12,'
        'VERTEBRAE_DISK_T10T11,VERTEBRAE_DISK_T9T10,VERTEBRAE_DISK_T8T9,VERTEBRAE_DISK_T7T8,VERTEBRAE_DISK_T6T7,VERTEBRAE_DISK_T5T6,VERTEBRAE_DISK_T4T5,'
        'VERTEBRAE_DISK_T3T4,VERTEBRAE_DISK_T2T3,VERTEBRAE_DISK_T1T2,VERTEBRAE_DISK_C7T1,VERTEBRAE_DISK_C6C7,VERTEBRAE_DISK_C5C6,VERTEBRAE_DISK_C4C5,'
        'VERTEBRAE_DISK_C3C4,VERTEBRAE_DISK_C2C3,'
        'SPINAL_CANAL',
        data='in_data',
        bundle='model',
        the='Spider baseline vertebrae segmentation for the input sagittal spine MRI or CT. Remapped segmentation output: '
            '0: background '
            '1-24: different vertebrae numbered from the bottom (i.e. L5 = 1, C1 = 24) (includes partially visible vertebrae from raw segmentation)'
            '25-48: different intervertebral discs (also numbered from the bottom, i.e. L5/S1 = 25, C2/C3 = 48) '
            '49: spinal canal'
    )
    def task(self, instance: Instance, in_data: InstanceData, out_data_raw: InstanceData, out_data: InstanceData) -> None:
        if in_data.type.meta <= MR:
            config = self.setup_mr()
        elif in_data.type.meta <= CT:
            config = self.setup_ct()
        else:
            raise ValueError(f"SpiderBaselineRunner does not support other modalities than CT and MR")
        self.process(
            in_data=in_data,
            out_data_raw=out_data_raw,
            **config
        )
        self.create_remapped_segmentation(out_data_raw, out_data)

    def create_remapped_segmentation(self, out_data_raw: InstanceData, out_data: InstanceData):
        # Create the remapping dictionary to reorder output labels so they will be picked up in the correct order by DicomSeg
        remap_dict = {i: i for i in range(0, 25)}  # keep labels 0-24 the same
        remap_dict.update({i: i - 100 for i in
                           range(101, 125)})  # partially visible vertebrae get remapped to regular vertebrae labels
        remap_dict.update({i:i-201+25 for i in range(201, 225)})  # remaps intervertebral discs to 25-48
        remap_dict.update({100: 49})                              # remaps spinal canal to 49

        # Convert the mapping to a 1d numpy vector by generating a value for each potential segmentation value
        # Each value is mapped to zero by default and the mapping values are overwritten by the remap_dict
        remap_np = np.zeros((226,), dtype=int)
        remap_np[list(remap_dict.keys())] = list(remap_dict.values())

        self.log(f"Remap generated segmentation output to: {out_data.abspath}", level="NOTICE")
        self.log(f"  mapping used: {remap_dict}", level="DEBUG")
        seg_sitk = SimpleITK.ReadImage(out_data_raw.abspath)
        seg_np = SimpleITK.GetArrayFromImage(seg_sitk)
        seg_remapped_np = remap_np[seg_np]  # actual remapping
        seg_remapped_sitk = SimpleITK.GetImageFromArray(seg_remapped_np)
        seg_remapped_sitk.CopyInformation(seg_sitk)
        SimpleITK.WriteImage(seg_remapped_sitk, out_data.abspath, True)

    def setup_mr(self) -> Dict:
        self.log(f"Setup the SPIDER-Baseline algorithm, using `MR` settings", level="NOTICE")
        self._create_input_metadata_file(modality="MR")
        return dict(
            surface_erosion_threshold = -2000,
            min_fragment_size = 100,
            min_size_cont = 100
        )

    def setup_ct(self) -> Dict:
        self.log(f"Setup the SPIDER-Baseline algorithm, using `CT` settings", level="NOTICE")
        self._create_input_metadata_file(modality="CT")
        return dict(
            surface_erosion_threshold = 200,
            min_fragment_size = 500,
            min_size_cont = 500
        )

    def process(self, in_data: InstanceData, out_data_raw: InstanceData, min_size_cont: int, min_fragment_size: int, surface_erosion_threshold: int):
        # Prepare input data
        in_data_path = in_data.abspath
        internal_img_path = SPIDER_INTERNAL_DATASET_PATH / "images" / "input_img.mha"
        self.log(f"Copying data files to internal SPIDER data structure: {in_data_path} -> {internal_img_path}",
                 level="NOTICE")
        shutil.copy(in_data_path, str(internal_img_path))
        self.log(f"Run the SPIDER-Baseline algorithm", level="NOTICE")
        # The algorithm is configured to run on an internal data folder structure and is
        # further configured by the JSON files in the Dockerfile
        # The algorithm parameters specify the following:
        # * The first argument specifies the folder to look for the model files
        # * The epoch specifies the model weights file to use, which should be something like: `999999.pt`
        # * The dataset is set to the internal SPIDER data structure
        # * The following parameters are set based on modality
        #   * surface_erosion_threshold
        #   * min_fragment_size
        #   * min_size_cont
        # * The original image dimensions are used to generate the output segmentation
        SpiderAlgorithm(
            [
                "SPIDER-Baseline",
                "--epoch", "999999",
                "--dataset", "spider_input",
                "--surface_erosion_threshold", str(surface_erosion_threshold),
                "--min_fragment_size", str(min_fragment_size),
                "--min_size_cont", str(min_size_cont),
                "--export_original"
            ]
        ).execute()
        # Export generated output segmentation
        source_output_file = SPIDER_INTERNAL_OUTPUT_DIR / "input_img_total_segmentation_original.mha"
        self.log(f"Move raw generated segmentation output from: {source_output_file} -> {out_data_raw.abspath}",
                 level="NOTICE")
        shutil.move(str(source_output_file), out_data_raw.abspath)
        # Run internal output cleanup
        shutil.rmtree(str(SPIDER_INTERNAL_OUTPUT_DIR))

    def __init__(self, config, local_config):
        # Create the arguments config json file on initialization
        # based on the traversal_direction_up configuration setting
        super().__init__(config=config, local_config=local_config)
        self._create_arguments_config_file(traversal_direction_up=self.traversal_direction_up)

    def _create_arguments_config_file(self, traversal_direction_up: bool = True):
        with open(SPIDER_EXPERIMENT_ARGUMENTS_FILE, "w") as f:
            json.dump({
                "filters": 50,  # Fixed for this model
                "traversal_direction": "up" if traversal_direction_up else "down"
                }, f
            )

    def _create_input_metadata_file(self, modality: str):
        # This configures the internal meta-data required for the input image
        # while running the algorithm. Should be set before running the algorithm
        with open(SPIDER_INTERNAL_DATASET_META_FILE, "w") as f:
            json.dump(
                [{
                    "identifier": "input_img",
                    "subset": "testing",
                    "filetype": "mha",  # only supports mha
                    "slice-order-reversed": False,
                    "rescale-intercept": 0,
                    "modality": modality,  # should be MR or CT
                    "ignore_slices_top": 0,
                    "ignore_slices_bottom": 0
                }]
            , f)
