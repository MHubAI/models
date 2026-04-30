# import mhub fw
from mhubio.core import Module, IO, Instance, InstanceData, ValueOutput, ClassOutput

# import pipeline
import os
import SimpleITK as sitk
from ..src.cacmapping import register_cac_mask 
from ..src.agatston import agatston_score_slices2, agatston_score_interpretation

@ValueOutput.Name('ags')
@ValueOutput.Label('AgatstonScore')
@ValueOutput.Type(int)
@ValueOutput.Description('Prediction of the agatson score.')
class AgatstonScore(ValueOutput):
   pass

@ValueOutput.Name('mags')
@ValueOutput.Label('MappedCAC AgatstonScore')
@ValueOutput.Type(int)
@ValueOutput.Description('Prediction of the agatson score using the mapped cac segmentation.')
class MappedAgatstonScore(ValueOutput):
   pass

@ClassOutput.Name('rc')
@ClassOutput.Label('RiskCategory')
@ClassOutput.Description('Prediction of the risk category.')
@ClassOutput.Class(0, 'No', the='The zero risk group for AGS equal zero.')
@ClassOutput.Class(1, 'Low', the='Class describing the lowest risk group.')
@ClassOutput.Class(2, 'Moderate', the='Moderate risk.')
@ClassOutput.Class(3, 'High', 'High risk.')
class RiskCategory(ClassOutput):
   pass

@ClassOutput.Name('mrc')
@ClassOutput.Label('MappedCAC RiskCategory')
@ClassOutput.Description('Prediction of the risk category based on the mapped cac segmentation.')
@ClassOutput.Class(0, 'No', the='The zero risk group for AGS equal zero.')
@ClassOutput.Class(1, 'Low', the='Class describing the lowest risk group.')
@ClassOutput.Class(2, 'Moderate', the='Moderate risk.')
@ClassOutput.Class(3, 'High', 'High risk.')
class MappedRiskCategory(ClassOutput):
   pass

class DeepCACPostProcessor(Module):

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('cac', 'nrrd:mod=seg:roi=CAC', the='detected coronary artery calcification')
    @IO.Output('mapped_cac', 'maped_pcac.nrrd', 'nrrd:mod=seg:variant=mapped', data='cac', the='delineation of combined structures with HU > 130 where overlapping with detected cac')
    @IO.OutputData('ags', AgatstonScore, data='cac', the='agatston score')
    @IO.OutputData('mags', MappedAgatstonScore, data='cac', the='agatston score')
    @IO.OutputData('risk', RiskCategory, data='cac', the='risk classification')
    @IO.OutputData('mrisk', MappedRiskCategory, data='cac', the='risk classification')
    def task(self, instance: Instance, image: InstanceData, cac: InstanceData, mapped_cac: InstanceData, ags: AgatstonScore, risk: RiskCategory, mags: MappedAgatstonScore, mrisk: MappedRiskCategory) -> InstanceData:
      
        # load img using sitk as x, y, z numpy array
        img_vol = sitk.ReadImage(image.abspath)
        img_np = sitk.GetArrayFromImage(img_vol).transpose(2, 1, 0)

        # load prediction using sitk
        cacpred_file = os.path.join(cac.abspath)
        assert os.path.exists(cacpred_file), f"no prediction found, expected {cacpred_file}"
        cacpred_vol = sitk.ReadImage(cacpred_file)
        cacpred_np = sitk.GetArrayFromImage(cacpred_vol).transpose(2, 1, 0)

        # mapping
        mcacpred_np = register_cac_mask(img_np, cacpred_np)

        # store mapped
        mcacpred_vol = sitk.GetImageFromArray(mcacpred_np.transpose(2, 1, 0)) # type: ignore
        mcacpred_vol.CopyInformation(img_vol)

        # save the prediction to the file system
        sitk.WriteImage(mcacpred_vol, mapped_cac.abspath)

        # calculate ags and risk for original cac prediction
        ags.value = round(agatston_score_slices2(cacpred_np, img_np, nrConPx=3, spacing=img_vol.GetSpacing()))
        risk.value = agatston_score_interpretation(ags.value)

        # calculate ags and risk for mapped cac prediction
        mags.value = round(agatston_score_slices2(mcacpred_np, img_np, nrConPx=3, spacing=img_vol.GetSpacing()))
        mrisk.value = agatston_score_interpretation(mags.value)

