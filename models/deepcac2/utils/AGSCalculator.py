# import mhub fw
from mhubio.core import Module, IO, Instance, InstanceData, ValueOutput, ClassOutput

# import pipeline
import os
import SimpleITK as sitk
from ..src.agatston import agatston_score_slices2, agatston_score_interpretation


@ValueOutput.Name('ags')
@ValueOutput.Label('AgatstonScore')
@ValueOutput.Type(int)
@ValueOutput.Description('Prediction of the agatson score.')
class AgatstonScore(ValueOutput):
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


@IO.ConfigInput('cac', 'nrrd:mod=seg:roi=CAC AND NOT any:variant=mapped',  the='input ct scan')
class AGSCalculator(Module):

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('cac', the='detected coronary artery calcification')
    @IO.OutputData('ags', AgatstonScore, data='cac', the='agatston score')
    @IO.OutputData('risk', RiskCategory, data='cac', the='risk classification')
    def task(self, instance: Instance, image: InstanceData, cac: InstanceData, ags: AgatstonScore, risk: RiskCategory) -> InstanceData:
    
        # load img using sitk as x, y, z numpy array
        img_vol = sitk.ReadImage(image.abspath)
        img_np = sitk.GetArrayFromImage(img_vol).transpose(2, 1, 0)

        # load prediction using sitk
        cacpred_file = cac.abspath
        assert os.path.exists(cacpred_file), f"no prediction found, expected {cacpred_file}"
        cacpred_vol = sitk.ReadImage(cacpred_file)
        cacpred_np = sitk.GetArrayFromImage(cacpred_vol).transpose(2, 1, 0)

        # calculate ags and risk
        ags.value = round(agatston_score_slices2(cacpred_np, img_np, nrConPx=3, spacing=img_vol.GetSpacing()))
        risk.value = agatston_score_interpretation(ags.value)
