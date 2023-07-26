"""
------------------------------------------------
Mhub / DIAG - CLI Run script for the TIGER LB2
------------------------------------------------

------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------
"""

import argparse
from pathlib import Path

import pipeline.tils_pipeline as tils_pipeline
import algorithm.rw as rw

import torch


def tiger_lb2_cli() -> None:
    parser = argparse.ArgumentParser("Tiger LB2 Run CLI")
    parser.add_argument("input_file", type=str, help="Input WSI TIFF file path")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    args = parser.parse_args()
    run_tiger_lb2(
        wsi_filepath=Path(args.input_file),
        output_json_file=Path(args.output_file)
    )


def run_tiger_lb2(wsi_filepath: Path, output_json_file: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("run_tiger_lb2 requires CUDA to be available!")

    print(f"Input WSI: {wsi_filepath}")
    wsi_mri = rw.open_multiresolutionimage_image(wsi_filepath)

    tils_score_writer = rw.TilsScoreWriter(output_json_file)
    tils_score = tils_pipeline.run_tils_pipeline(wsi_mri)

    print(f"Writing tils score to {output_json_file}")
    tils_score_writer.set_tils_score(tils_score=tils_score)
    tils_score_writer.save()


if __name__ == "__main__":
    tiger_lb2_cli()
