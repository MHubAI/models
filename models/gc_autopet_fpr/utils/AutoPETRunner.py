"""
-----------------------------------------------------------------------
Mhub / DIAG - Run Module for AutoPET false positive reduction algorithm
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""
from typing import Tuple

import SimpleITK
from mhubio.core import Instance, DataTypeQuery, FileType, InstanceData, IO, Module, Meta

from pathlib import Path

# Import AutoPET challenge algorithm installed from the /YigePeng/AutoPET_False_Positive_Reduction repository
from process import Hybrid_cnn as AutoPETAlgorithm


class AutoPETRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'mha|nifti:mod=ct:resampled=true', the='input FDG CT scan, resampled to FDG PET scan')
    @IO.Input('in_data_pet', 'mha|nifti:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data', 'tumor_segmentation.mha', 'mha:mod=seg:model=AutoPET:roi=NEOPLASM_MALIGNANT_PRIMARY', bundle='model', the='predicted tumor segmentation within the input FDG PET/CT scan')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data: InstanceData) -> None:
        # Instantiate the algorithm and check GPU availability
        algorithm = AutoPETAlgorithm()
        algorithm.check_gpu()

        # Define some paths which are used internally by the algorithm
        internal_ct_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0001.nii.gz'
        internal_pet_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0000.nii.gz'
        internal_output_nifti_file = Path(algorithm.result_path) / algorithm.nii_seg_file

        self.log("Prepare input data", level="NOTICE")
        def prepare_input_data(input_data: InstanceData, internal_target_file: Path) -> Tuple[int, int ,int]:
            sitk_img = SimpleITK.ReadImage(input_data.abspath)
            original_size = sitk_img.GetSize()
            # resample x and y dimensions to 400 as expected by algorithm input
            sitk_img = self._resample_input(
                input_image=sitk_img,
                desired_output_size=(400, 400, sitk_img.GetSize()[2])
            )
            SimpleITK.WriteImage(sitk_img, str(internal_target_file), True)
            return original_size

        osize_ct = prepare_input_data(in_data_ct, internal_ct_nifti_file)
        osize_pt = prepare_input_data(in_data_pet, internal_pet_nifti_file)
        assert osize_ct == osize_pt, f"Input PET and CT should have same dimensions, found PET {osize_pt} and CT {osize_ct}"

        self.log("Run AutoPET FPR algorithm", level="NOTICE")
        algorithm.predict_ssl()

        self.log(f"Convert output nii segmentation to mha output (resample to original size if necessary): {internal_output_nifti_file} -> {out_data.abspath}", level="NOTICE")
        output_sitk = SimpleITK.ReadImage(str(internal_output_nifti_file))
        output_sitk = self._resample_input(
            input_image=output_sitk,
            desired_output_size=osize_ct,
            interpolation=SimpleITK.sitkNearestNeighbor
        )
        SimpleITK.WriteImage(output_sitk, out_data.abspath, True)


    def _resample_input(
        self,
        input_image: SimpleITK.Image,
        desired_output_size: Tuple[float, float, float],  # x y z
        interpolation: int = SimpleITK.sitkBSplineResamplerOrder1,
    ) -> SimpleITK.Image:
        if input_image.GetSize() != desired_output_size:
            output_spacing = [spacing / (out_size / size) for spacing, size, out_size in zip(input_image.GetSpacing(), input_image.GetSize(), desired_output_size)]
            output_origin = [in_origin + 0.5 * (out_spacing - in_spacing) for in_origin, out_spacing, in_spacing in zip(input_image.GetOrigin(), output_spacing, input_image.GetSpacing())]
            self.log(
                f"Resampling input image with shape: {input_image.GetSize()} -> {desired_output_size} and spacing: "
                f"{input_image.GetSpacing()} -> {output_spacing}", level="NOTICE"
            )
            resample_filter = SimpleITK.ResampleImageFilter()
            resample_filter.SetInterpolator(interpolation)
            resample_filter.SetDefaultPixelValue(-1024)
            resample_filter.SetSize(desired_output_size)
            resample_filter.SetOutputSpacing(output_spacing)
            resample_filter.SetOutputDirection(input_image.GetDirection())
            resample_filter.SetOutputOrigin(output_origin)
            return resample_filter.Execute(input_image)
        return input_image
