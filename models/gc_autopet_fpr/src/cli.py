"""
-----------------------------------------------------------------------
GC / MHub - CLI for the GC AutoPET FPR Algorithm
  The model algorith was wrapped in a CLI to ensure
  the mhub framework is able to properly capture the nnUNet
  stdout/stderr outputs
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""
import argparse
from typing import Tuple
from pathlib import Path

import SimpleITK

# Import AutoPET challenge algorithm installed from the /YigePeng/AutoPET_False_Positive_Reduction repository
from process import Hybrid_cnn as AutoPETAlgorithm


def process_fdg_pet_ct(in_data_ct: Path, in_data_pet: Path, out_data: Path) -> None:
    # Instantiate the algorithm and check GPU availability
    algorithm = AutoPETAlgorithm()
    algorithm.check_gpu()

    # Define some paths which are used internally by the algorithm
    internal_ct_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0001.nii.gz'
    internal_pet_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0000.nii.gz'
    internal_output_nifti_file = Path(algorithm.result_path) / algorithm.nii_seg_file

    print("Prepare input data")
    def prepare_input_data(input_data: Path, internal_target_file: Path) -> Tuple[int, int ,int]:
        sitk_img = SimpleITK.ReadImage(str(input_data))
        original_size = sitk_img.GetSize()
        # resample x and y dimensions to 400 as expected by algorithm input
        sitk_img = maybe_resample_input(
            input_image=sitk_img,
            desired_output_size=(400, 400, sitk_img.GetSize()[2])
        )
        SimpleITK.WriteImage(sitk_img, str(internal_target_file), True)
        return original_size

    osize_ct = prepare_input_data(in_data_ct, internal_ct_nifti_file)
    osize_pt = prepare_input_data(in_data_pet, internal_pet_nifti_file)
    assert osize_ct == osize_pt, f"Input PET and CT should have same dimensions, found PET {osize_pt} and CT {osize_ct}"

    print("Run AutoPET FPR algorithm")
    algorithm.predict_ssl()

    print(f"Convert output nii segmentation to mha output (resample to original size if necessary): {internal_output_nifti_file} -> {out_data}")
    output_sitk = SimpleITK.ReadImage(str(internal_output_nifti_file))
    output_sitk = maybe_resample_input(
        input_image=output_sitk,
        desired_output_size=osize_ct,
        interpolation=SimpleITK.sitkNearestNeighbor
    )
    SimpleITK.WriteImage(output_sitk, str(out_data), True)


def maybe_resample_input(
    input_image: SimpleITK.Image,
    desired_output_size: Tuple[float, float, float],  # x y z
    interpolation: int = SimpleITK.sitkBSplineResamplerOrder1,
) -> SimpleITK.Image:
    if input_image.GetSize() != desired_output_size:
        output_spacing = [spacing / (out_size / size) for spacing, size, out_size in zip(input_image.GetSpacing(), input_image.GetSize(), desired_output_size)]
        output_origin = [in_origin + 0.5 * (out_spacing - in_spacing) for in_origin, out_spacing, in_spacing in zip(input_image.GetOrigin(), output_spacing, input_image.GetSpacing())]
        print(
            f"Resampling input image with shape: {input_image.GetSize()} -> {desired_output_size} and spacing: "
            f"{input_image.GetSpacing()} -> {output_spacing}"
        )
        resample_filter = SimpleITK.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolation)
        resample_filter.SetSize(desired_output_size)
        resample_filter.SetOutputSpacing(output_spacing)
        resample_filter.SetOutputDirection(input_image.GetDirection())
        resample_filter.SetOutputOrigin(output_origin)
        return resample_filter.Execute(input_image)
    return input_image


def autopet_fpr_cli():
    parser = argparse.ArgumentParser(
        "CLI for running the AutoPET FPR algorithm on FDG PET-CT pair of input "
        "images to generate a tumor segmentation map"
    )
    parser.add_argument("in_data_pet", type=str, help="input FDG PET scan (MHA/NIFTI)")
    parser.add_argument("in_data_ct", type=str, help="input FDG CT scan, resampled to FDG PET scan (MHA/NIFTI)")
    parser.add_argument("out_data", type=str, help="predicted tumor segmentation within the input FDG PET/CT scan (MHA)")
    args = parser.parse_args()
    process_fdg_pet_ct(
        in_data_pet=Path(args.in_data_pet),
        in_data_ct=Path(args.in_data_ct),
        out_data=Path(args.out_data),
    )


if __name__ == "__main__":
    autopet_fpr_cli()
