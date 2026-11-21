"""Match the image-axis convention used to train the DeepCAC2 CAC model."""

import SimpleITK as sitk
from mhubio.core import IO, Instance, InstanceData, Module


@IO.Config("delete_input_file", bool, False, the="Delete the replaced NIfTI file after successful LPS normalization")
class LPSOrienter(Module):
    """Create an LPS working image before either segmentation model runs."""

    delete_input_file: bool

    @IO.Instance()
    @IO.Input("image", "nifti:mod=ct")
    @IO.Output(
        "oriented_image", "image_lps.nii.gz", "nifti:mod=ct:orientation=LPS",
        data="image", bundle="lps",
    )
    def task(self, instance: Instance, image: InstanceData, oriented_image: InstanceData) -> None:
        ct = sitk.ReadImage(image.abspath)
        if ct.GetDimension() != 3 or ct.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("LPS normalization requires a scalar 3D CT image.")

        # DICOMOrient permutes/flips voxel axes without interpolating HU values
        # or changing physical voxel locations. Oblique directions are retained.
        normalized = sitk.DICOMOrient(ct, "LPS")
        sitk.WriteImage(normalized, oriented_image.abspath)

        # Both models and the scorer must select this working image. Retire the
        # previous entry; delete its file only when the workflow opts in to
        # cleaning generated intermediates. Imported NIfTI files are kept by default.
        instance.data.remove(image, delete_files=self.delete_input_file)
