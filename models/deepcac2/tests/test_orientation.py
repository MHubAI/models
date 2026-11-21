"""Verify anatomical alignment across image-axis conventions and MHub routing."""

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

pytest.importorskip("mhubio")
from mhubio.core import Config, DataType, Instance, InstanceData
from models.deepcac2.utils.LPSOrienter import LPSOrienter
from models.deepcac2.utils.CalciumAnalysisCalculator import CalciumAnalysisCalculator


ORIENTATIONS = [
    "".join(axes)
    for pairs in itertools.permutations((("L", "R"), ("P", "A"), ("S", "I")))
    for axes in itertools.product(*pairs)
]


def _reference():
    values = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) * 17.25 - 1024
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((0.625, 0.8, 2.5))
    image.SetOrigin((-123.125, 45.5, -201.25))
    return image


def _normalize(tmp_path, monkeypatch, image, delete_input_file=None):
    monkeypatch.setenv("MLOG_STOP_ON_ERROR", "YES")
    settings = {"general": {"data_base_dir": str(tmp_path)}}
    if delete_input_file is not None:
        settings["modules"] = {"LPSOrienter": {"delete_input_file": delete_input_file}}
    config = Config(config=settings, args=False)
    instance = Instance("working")
    config.data.addInstance(instance)
    source = tmp_path / "input.nii.gz"
    sitk.WriteImage(image, str(source))
    source.chmod(0o444)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = InstanceData(str(source), DataType.fromString("nifti:mod=ct:pid=TEST"), instance=instance)
    original.confirm()
    LPSOrienter(config).task()
    if delete_input_file:
        assert not source.exists()
    else:
        assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
    active = list(instance.data.filter("nifti:mod=ct"))
    assert len(active) == 1 and active[0] is not original
    assert active[0].confirmed
    assert active[0].type.meta["pid"] == "TEST"
    assert active[0].type.meta["orientation"] == "LPS"
    assert Path(active[0].abspath).is_relative_to(tmp_path / "working")
    return config, instance, sitk.ReadImage(active[0].abspath)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_axes_and_voxel_locations_match_training_convention(tmp_path, monkeypatch, orientation):
    reference = _reference()
    source = sitk.DICOMOrient(reference, orientation)
    _, _, actual = _normalize(tmp_path, monkeypatch, source)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(reference))
    assert actual.GetSize() == reference.GetSize()
    np.testing.assert_allclose(actual.GetSpacing(), reference.GetSpacing(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual.GetDirection(), reference.GetDirection(), atol=1e-6, rtol=0)
    for index in [(0, 0, 0), (5, 4, 3), (2, 3, 1)]:
        np.testing.assert_allclose(
            actual.TransformIndexToPhysicalPoint(index),
            reference.TransformIndexToPhysicalPoint(index), atol=2e-5, rtol=0,
        )


def test_obliquity_is_preserved_without_interpolation(tmp_path, monkeypatch):
    reference = _reference()
    theta = np.deg2rad(7)
    direction = (np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta))
    reference.SetDirection(direction)
    source = sitk.DICOMOrient(reference, "ASR")
    _, _, actual = _normalize(tmp_path, monkeypatch, source)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(reference))
    np.testing.assert_allclose(actual.GetDirection(), direction, atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual.GetOrigin(), reference.GetOrigin(), atol=2e-5, rtol=0)


def test_normalization_is_idempotent(tmp_path, monkeypatch):
    config, instance, first = _normalize(tmp_path, monkeypatch, sitk.DICOMOrient(_reference(), "RAS"))
    LPSOrienter(config).task()
    images = list(instance.data.filter("nifti:mod=ct"))
    assert len(images) == 1 and images[0].confirmed
    second = sitk.ReadImage(images[0].abspath)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(first), sitk.GetArrayFromImage(second))
    assert second.GetDirection() == first.GetDirection()
    assert second.GetOrigin() == first.GetOrigin()


@pytest.mark.parametrize("delete_input_file", [False, True])
def test_cleanup_option_preserves_the_normalized_image(tmp_path, monkeypatch, delete_input_file):
    reference = _reference()
    _, _, actual = _normalize(tmp_path, monkeypatch, sitk.DICOMOrient(reference, "LAS"), delete_input_file)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(actual), sitk.GetArrayFromImage(reference))


def test_cleanup_keeps_input_when_output_write_fails(tmp_path, monkeypatch):
    write_image = sitk.WriteImage

    def fail_normalized_write(image, filename, *args, **kwargs):
        if Path(filename).name == "image_lps.nii.gz":
            raise RuntimeError("Simulated output write failure")
        return write_image(image, filename, *args, **kwargs)

    monkeypatch.setattr(sitk, "WriteImage", fail_normalized_write)
    with pytest.raises(RuntimeError, match="Simulated output write failure"):
        _normalize(tmp_path, monkeypatch, _reference(), delete_input_file=True)
    source = tmp_path / "input.nii.gz"
    assert source.is_file()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(source))),
                                  sitk.GetArrayFromImage(_reference()))


@pytest.mark.parametrize("shape,is_vector", [((4, 5), False), ((2, 3, 4, 5), False), ((2, 3, 4, 2), True)])
def test_invalid_dimensions_or_vector_images_are_rejected(tmp_path, monkeypatch, shape, is_vector):
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32), isVector=is_vector)
    with pytest.raises(ValueError, match="scalar 3D CT"):
        _normalize(tmp_path, monkeypatch, image)
    assert (tmp_path / "input.nii.gz").is_file()


def test_scorer_uses_normalized_ct_with_an_aligned_mask(tmp_path, monkeypatch):
    reference = sitk.GetImageFromArray(np.full((2, 4, 5), 400, dtype=np.float32))
    reference.SetSpacing((0.5, 0.5, 3.0))
    reference.SetOrigin((-50, 20, -100))
    config, instance, ct = _normalize(tmp_path, monkeypatch, sitk.DICOMOrient(reference, "LAS"))
    values = np.zeros((2, 4, 5), dtype=np.uint8)
    values[:, 1:3, 1:3] = 1
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(ct)
    data = InstanceData("cac.nrrd", DataType.fromString(
        "nrrd:mod=seg:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION"), instance=instance)
    sitk.WriteImage(mask, data.abspath)
    data.confirm()
    CalciumAnalysisCalculator(config).task()
    assert instance.outputData.filter("cacs").first().value == 8.0
    assert instance.outputData.filter("calcium_volume").first().value == 6.0


@pytest.mark.parametrize("workflow", ["default", "dicom_dicom_plastimatch", "dicom_extended", "nifti_pid"])
def test_workflow_preprocessing_before_both_models(workflow):
    import yaml
    settings = yaml.safe_load((Path(__file__).parents[1] / "config" / f"{workflow}.yml").read_text())
    steps = settings["execute"]
    if workflow == "dicom_dicom_plastimatch":
        # This workflow uses the default converter without an orientation step.
        assert "LPSOrienter" not in steps
        assert steps.index("NiftiConverter") + 1 == steps.index("NNUnetRunner")
        assert steps.index("NNUnetRunner") < steps.index("DeepCACRunner") < steps.index("CalciumAnalysisCalculator")
        assert "engine" not in settings["modules"].get("NiftiConverter", {})
        return
    assert steps.count("LPSOrienter") == 1
    index = steps.index("LPSOrienter")
    assert steps[index + 1] == "NNUnetRunner"
    assert index < steps.index("DeepCACRunner") < steps.index("CalciumAnalysisCalculator")
    if workflow == "nifti_pid":
        assert steps[index - 1] == "FileStructureImporter"
        assert not settings["modules"].get("LPSOrienter", {}).get("delete_input_file", False)
    else:
        assert steps[index - 1] == "NiftiConverter"
        assert settings["modules"]["NiftiConverter"]["engine"] == "dcm2niix"
        assert settings["modules"]["LPSOrienter"]["delete_input_file"] is True
