"""
-----------------------------------------------------------
GC / MHub - CLI Run script for the TIGER LB2 Algorithm
  The model algorith was wrapped in a CLI to ensure
  the mhub framework is able to properly capture the nnUNet
  stdout/stderr outputs. Furthermore, it simplifies running
  the algorithm in its own environment using pipenv.
-----------------------------------------------------------

-----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------
"""
import argparse
from pathlib import Path

import torch
import SimpleITK

# The required pipeline methods are imported from the tiger_challenge repository
# The algorithm.rw module is imported for IO operations
import pipeline.tils_pipeline as tils_pipeline
import algorithm.rw as rw


def tiger_lb2_cli() -> None:
    parser = argparse.ArgumentParser("Tiger LB2 Run CLI")
    parser.add_argument("input_file", type=str, help="Input WSI TIFF file path")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("output_segmentation_file", type=str, help="Output segmentation MHA file path")
    args = parser.parse_args()
    run_tiger_lb2(
        wsi_filepath=Path(args.input_file),
        output_json_file=Path(args.output_file),
        output_segmentation_file=Path(args.output_segmentation_file),
    )


def run_tiger_lb2(wsi_filepath: Path, output_json_file: Path, output_segmentation_file: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("run_tiger_lb2 requires CUDA to be available!")

    print(f"Input WSI: {wsi_filepath}")
    wsi_mri = rw.open_multiresolutionimage_image(wsi_filepath)

    print("Run pipeline")
    tils_score, seg_mask_np = tils_pipeline.run_tils_pipeline(wsi_mri)

    print(f"Writing segmentation map to {output_segmentation_file}")
    seg_mask_sitk = SimpleITK.GetImageFromArray(seg_mask_np)
    SimpleITK.WriteImage(seg_mask_sitk, str(output_segmentation_file), True)

    print(f"Writing tils score to {output_json_file}")
    tils_score_writer = rw.TilsScoreWriter(output_json_file)
    tils_score_writer.set_tils_score(tils_score=tils_score)
    tils_score_writer.save()


if __name__ == "__main__":
    tiger_lb2_cli()
