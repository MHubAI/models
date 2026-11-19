"""MHub inference adapter for the DeepCAC2 plaque segmentation stage."""

from mhubio.core import Module, IO, Instance, InstanceData

from ..src.inference import run_inference
from ..src.subject import Sample


class DeepCACRunner(Module):

    # MHub resolves the CT and nnU-Net heart mask within each instance. The
    # SegDB ROI identifier is carried forward into DICOM SEG conversion.
    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('heart', 'nifti:mod=seg', the='input heart segmentation')    
    @IO.Output('cac', 'pcac.nrrd', 'nrrd:mod=seg:model=DeepCAC2:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION', the='detected coronary artery calcification')
    def task(self, instance: Instance, image: InstanceData, heart: InstanceData, cac: InstanceData) -> InstanceData:
        # Adapt MHub-managed paths to the small dictionary interface expected by
        # the original DeepCAC2 inference implementation.
        sample: Sample = {
            'id': instance.attr['id'],
            'img_path': image.abspath,
            'hrt_path': heart.abspath,
            'meta': {}
        }

        # The inference function writes directly to the output path allocated by
        # the IO.Output decorator; MHub confirms it after successful execution.
        run_inference(sample, cac.abspath)
