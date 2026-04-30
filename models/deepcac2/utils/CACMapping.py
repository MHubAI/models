# import mhub fw
from mhubio.core import Module, IO, Instance, InstanceData

# import pipeline
import os
import SimpleITK as sitk
from ..src.cacmapping import register_cac_mask

class CACMapping(Module):

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('cac', 'nrrd:mod=seg:roi=CAC', the='detected coronary artery calcification')
    @IO.Output('mapped_cac', 'maped_pcac.nrrd', 'nrrd:mod=seg:variant=mapped', data='cac', the='delineation of combined structures with HU > 130 where overlapping with detected cac')
    def task(self, instance: Instance, image: InstanceData, cac: InstanceData, mapped_cac: InstanceData) -> InstanceData:
      
        # load img using sitk as x, y, z numpy array
        img_vol = sitk.ReadImage(image.abspath)
        img_np = sitk.GetArrayFromImage(img_vol).transpose(2, 1, 0)

        # load prediction using sitk
        cacpred_file = os.path.join(cac.abspath)
        assert os.path.exists(cacpred_file), f"no prediction found, expected {cacpred_file}"
        cacpred_vol = sitk.ReadImage(cacpred_file)
        cacpred_np = sitk.GetArrayFromImage(cacpred_vol).transpose(2, 1, 0)

        # mapping
        rcacpred_np = register_cac_mask(img_np, cacpred_np)

        # store mapped
        rcacpred_vol = sitk.GetImageFromArray(rcacpred_np.transpose(2, 1, 0)) # type: ignore
        rcacpred_vol.CopyInformation(img_vol)

        # save the prediction to the file system
        sitk.WriteImage(rcacpred_vol, mapped_cac.abspath)

