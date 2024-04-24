"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for resampling FDG CT to the FDG-PET
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""

import nibabel
import nilearn.image

from mhubio.core import Instance, DataTypeQuery, InstanceData, IO, Module, Meta


class FDGPETCTResampleProcessor(Module):

    @IO.Instance()
    @IO.Input('in_data_ct', 'nifti:mod=ct', the='input FDG CT scan')
    @IO.Input('in_data_pet', 'nifti:mod=pt', the='input FDG PET scan')
    @IO.Output('out_data_ct', 'CTres.nii.gz', 'nifti:mod=ct:resampled=true', bundle='registered', the='Converted CT image resampled to PET image resolution and size')
    def task(self, instance: Instance, in_data_ct: InstanceData, in_data_pet: InstanceData, out_data_ct: InstanceData) -> None:
        # Convert PET and CT data to Nifti
        ct   = nibabel.load(in_data_ct.abspath)
        pet  = nibabel.load(in_data_pet.abspath)
        # Register and Resample CT to PET
        ct_res = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
        # Write output
        nibabel.save(ct_res, out_data_ct.abspath)
