"""
------------------------------------------------------------------------------------
Mhub / DIAG - Run Module for preprocessing the TCIA FDG-PET CT Lesions DICOM dataset
------------------------------------------------------------------------------------

------------------------------------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
------------------------------------------------------------------------------------
"""

import tempfile
from pathlib import Path

from mhubio.core import Instance, DataTypeQuery, InstanceData, IO, Module, Meta

# Import conversion routines from https://github.com/lab-midas/TCIA_processing
from tcia_dicom_to_nifti import dcm2nii_CT, dcm2nii_PET, resample_ct


class TciaDicomProcessor(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'dicom:mod=ct', the='input FDG CT scan')
    @IO.Input('in_data_pet', 'dicom:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data_pet', 'PET.nii.gz', 'nifti:mod=pt', bundle='tcia', the='Converted PET image')
    @IO.Output('out_data_ct', 'CTres.nii.gz', 'nifti:mod=ct', bundle='tcia', the='Converted CT image resampled to PET image resolution and size')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data_pet: InstanceData, out_data_ct: InstanceData) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Convert PET and CT data to Nifti
            dcm2nii_CT(Path(in_data_ct.abspath), tmp)
            dcm2nii_PET(Path(in_data_pet.abspath), tmp)
            # Resample/register CT to PET
            resample_ct(tmp)
            # Extracting converted files to expected outputs
            (tmp / "PET.nii.gz").replace(Path(out_data_pet.abspath))
            (tmp / "CTres.nii.gz").replace(Path(out_data_ct.abspath))
