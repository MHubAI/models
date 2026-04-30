# import mhub fw
from mhubio.core import Module, IO, Instance, InstanceData

# import pipeline
from ..src.inference import run_inference
from ..src.subject import Sample

class DeepCACRunner(Module):

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('heart', 'nifti:mod=seg', the='input heart segmentation')    
    @IO.Output('cac', 'pcac.nrrd', 'nrrd:mod=seg:model=DeepCAC2:roi=CAC', the='detected coronary artery calcification')
    def task(self, instance: Instance,  image: InstanceData, heart: InstanceData, cac: InstanceData) -> InstanceData:
        

        # create sample
        sample: Sample = {
            'id': instance.attr['id'],
            'img_path': image.abspath,
            'hrt_path': heart.abspath,
            'meta': {}
        }

        # run pipeline
        run_inference(sample, cac.abspath)

