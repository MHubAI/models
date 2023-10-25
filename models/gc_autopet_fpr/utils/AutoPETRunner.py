"""
-----------------------------------------------------------------------
Mhub / DIAG - Run Module for AutoPET false positive reduction algorithm
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""

from mhubio.core import Instance, DataTypeQuery, InstanceData, IO, Module, Meta

from pathlib import Path

# Import AutoPET challenge algorithm installed from the /YigePeng/AutoPET_False_Positive_Reduction repository
from process import Hybrid_cnn as AutoPETAlgorithm


# TODO should be moved to mhubio/core/templates.py
PT = Meta(mod="PT")  # Positron emission tomography (PET)


class AutoPETRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'mha:mod=ct', the='input FDG CT scan')
    @IO.Input('in_data_pet', 'mha:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data', 'tumor_segmenation.mha', 'mha:mod=seg:model=AutoPET:roi=NEOPLASM_MALIGNANT_PRIMARY', bundle='model', the='predicted tumor segmentation within the input FDG PET/CT scan')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data: InstanceData) -> None:
        # Define some paths which are used internally by the algorithm
        internal_ct_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0001.nii.gz'
        internal_pet_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0000.nii.gz'
        internal_output_nifti_file = Path(algorithm.result_path) / algorithm.nii_seg_file

        # Instantiate the algorithm and check GPU availability
        algorithm = AutoPETAlgorithm()
        algorithm.check_gpu()

        self.v(" > Prepare input data")
        algorithm.convert_mha_to_nii(in_data_ct.abspath, str(internal_ct_nifti_file))
        algorithm.convert_mha_to_nii(in_data_pet.abspath, str(internal_pet_nifti_file))

        self.v(" > Run AutoPET FPR algorithm")
        algorithm.predict_ssl()

        self.v(f" > Convert output nii segmentation to mha output: {internal_output_nifti_file} -> {out_data.abspath}")
        algorithm.convert_nii_to_mha(str(internal_output_nifti_file), out_data.abspath)
