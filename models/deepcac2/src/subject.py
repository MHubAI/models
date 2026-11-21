from typing import Tuple, TypedDict, Dict, Any, Optional
import torchio as tio

class Sample(TypedDict):
    id: str
    img_path: str
    hrt_path: Optional[str]
    meta: Dict[str, Any]

def get_subject(sample: Sample) -> tio.Subject:
    return tio.Subject(
        image=tio.ScalarImage(sample['img_path']),
        heart=tio.LabelMap(sample['hrt_path'])
    )

def get_tfx(spacing: Tuple[float, float, float], hbb: Tuple[int, int, int], apply_znorm: bool) -> tio.Transform:

    # resample
    resample_tfx = tio.Resample(
        target = spacing # x, y, z
    )

    # cropping (105, 184, 212) -> (184, 184, 212)
    crop_tfx = tio.CropOrPad(
        target_shape = hbb,
        mask_name = 'heart'
    )

    # windowing (apply abdomen soft tissue window: W:350, L:40)
    clamp_tfx = tio.Clamp(
        out_min = -135, 
        out_max = 500,
        keep = {'image': 'image_original_hu'}
    )

    # per patient z-norm of HU values
    znorm_tfx = tio.ZNormalization(
        exclude = ['image_original_hu']
    )

    # map values to [-1, 1]
    intmap_tfx = tio.RescaleIntensity(
        out_min_max = (0, 1),
        exclude = ['image_original_hu']
    )

    # composed transformatiuon chain
    return tio.Compose([
        resample_tfx,
        crop_tfx,
        clamp_tfx,
        *([znorm_tfx] if apply_znorm else []),
        intmap_tfx
    ])
