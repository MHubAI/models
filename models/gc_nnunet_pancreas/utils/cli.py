"""
----------------------------------------------------
GC / MHub - CLI for the GC nnUnet Pancreas Algorithm
----------------------------------------------------

----------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
----------------------------------------------------
"""
import argparse
from pathlib import Path

# Import the algorithm pipeline class from the CE-CT_PDAC_AutomaticDetection_nnUnet repository
from process import PDACDetectionContainer


def run_pdac_detection(
    input_ct_image: Path, output_heatmap: Path, output_segmentation: Path
):
    # Configure the algorithm pipeline class and run it
    algorithm = PDACDetectionContainer()
    algorithm.ct_image = str(input_ct_image)  # set as str not Path
    algorithm.heatmap = output_heatmap
    algorithm.segmentation = output_segmentation
    algorithm.process()


def run_pdac_detection_cli():
    parser = argparse.ArgumentParser("CLI for the GC nnUNet Pancreas Algorithm")
    parser.add_argument(
        "input_ct_image",
        type=str,
        help="input CT scan (MHA)"
    )
    parser.add_argument(
        "output_heatmap",
        type=str,
        help="heatmap of the pancreatic tumor likelihood (MHA)",
    )
    parser.add_argument(
        "output_segmentation",
        type=str,
        help="segmentation map of the pancreas (MHA), with the following classes: "
        "0-background, 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, "
        "6-cysts, 7-renal vein",
    )
    args = parser.parse_args()
    run_pdac_detection(
        input_ct_image=Path(args.input_ct_image),
        output_heatmap=Path(args.output_heatmap),
        output_segmentation=Path(args.output_segmentation),
    )


if __name__ == "__main__":
    run_pdac_detection_cli()
