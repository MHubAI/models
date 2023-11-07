"""
-----------------------------------------------------------------------
Mhub / DIAG - Run Module for AutoPET false positive reduction algorithm
-----------------------------------------------------------------------

-----------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------------------
"""

from mhubio.core import Instance, DataTypeQuery, FileType, InstanceData, IO, Module, Meta

import shutil
from pathlib import Path

# Import AutoPET challenge algorithm installed from the /YigePeng/AutoPET_False_Positive_Reduction repository
from process import Hybrid_cnn as AutoPETAlgorithm


# TODO should be moved to mhubio/core/templates.py
PT = Meta(mod="PT")  # Positron emission tomography (PET)


class AutoPETRunner(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'mha|nifti:mod=ct', the='input FDG CT scan')
    @IO.Input('in_data_pet', 'mha|nifti:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data', 'tumor_segmenation.mha', 'mha:mod=seg:model=AutoPET:roi=NEOPLASM_MALIGNANT_PRIMARY', bundle='model', the='predicted tumor segmentation within the input FDG PET/CT scan')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data: InstanceData) -> None:
        # Instantiate the algorithm and check GPU availability
        algorithm = AutoPETAlgorithm()
        algorithm.check_gpu()

        # Define some paths which are used internally by the algorithm
        internal_ct_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0001.nii.gz'
        internal_pet_nifti_file = Path(algorithm.nii_path) / 'TCIA_001_0000.nii.gz'
        internal_output_nifti_file = Path(algorithm.result_path) / algorithm.nii_seg_file

        self.log(" > Prepare input data", level="NOTICE")
        def prepare_input_data(input_data: InstanceData, internal_target_file: Path):
            if input_data.type.ftype == FileType.NIFTI:
                shutil.copy(input_data.abspath, str(internal_target_file))
            elif input_data.type.ftype == FileType.MHA:
                algorithm.convert_mha_to_nii(input_data.abspath, str(internal_target_file))
            else:
                raise NotImplementedError("prepare_input_data expects NIFTI or MHA as input")


        prepare_input_data(in_data_ct, internal_ct_nifti_file)
        prepare_input_data(in_data_pet, internal_pet_nifti_file)

        self.log(" > Run AutoPET FPR algorithm", level="NOTICE")
        algorithm.predict_ssl()

        self.log(f" > Convert output nii segmentation to mha output: {internal_output_nifti_file} -> {out_data.abspath}", level="NOTICE")
        algorithm.convert_nii_to_mha(str(internal_output_nifti_file), out_data.abspath)
