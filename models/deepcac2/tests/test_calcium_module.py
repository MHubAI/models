"""Exercise the MHub adapter against real image files and output schemas."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

pytest.importorskip("mhubio")
from mhubio.core import Instance, Meta, OutputDataCollection
from models.deepcac2.src.scoring import analyze_calcium, SCORING_VERSION
from models.deepcac2.utils.CalciumAnalysisCalculator import (
    CalciumAnalysisCalculator, Cacs, CalciumVolume, LesionCount, SegmentationVolume, RiskCategory, ScoringVersion, LesionConnectivity,
)


def _run(tmp_path, spacing=(1.0, 1.0, 3.0), mismatch=None, hu=100.4, connectivity=26):
    ct = sitk.GetImageFromArray(np.full((1, 2, 2), hu, dtype=float))
    ct.SetSpacing(spacing)
    mask = sitk.GetImageFromArray(np.ones((1, 2, 2), dtype=np.uint8))
    mask.CopyInformation(ct)
    if mismatch:
        getattr(mask, "Set" + mismatch[0])(mismatch[1])
    ct_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "mask.nrrd"
    sitk.WriteImage(ct, str(ct_path))
    sitk.WriteImage(mask, str(mask_path))
    outputs = [Cacs(), RiskCategory(), CalciumVolume(), SegmentationVolume(), LesionCount(), ScoringVersion(), LesionConnectivity()]
    module = CalciumAnalysisCalculator(SimpleNamespace(logger=None), local_config={"lesion_connectivity": connectivity})
    CalciumAnalysisCalculator.task._mhubio_ofunc(
        module, Instance(), SimpleNamespace(abspath=str(ct_path)),
        SimpleNamespace(abspath=str(mask_path), type=SimpleNamespace(meta=Meta(mod="seg"))),
        *outputs,
        *[OutputDataCollection() for _ in range(6)],
    )
    loaded_ct = sitk.ReadImage(str(ct_path))
    result = analyze_calcium(sitk.GetArrayFromImage(loaded_ct), sitk.GetArrayFromImage(mask), loaded_ct.GetSpacing(), lesion_connectivity=connectivity)
    return result, outputs


def test_nonempty_mask_with_no_eligible_calcium(tmp_path):
    result, outputs = _run(tmp_path)
    assert result["cacs"] == 0
    assert result["calcium_volume_mm3"] == 0
    assert result["segmentation_volume_mm3"] == 12
    assert result["lesions"] == []
    assert [o.value for o in outputs[:5]] == [0.0, 0, 0.0, 12.0, 0]
    assert outputs[5].value == SCORING_VERSION


def test_float_score_and_risk_are_preserved(tmp_path):
    result, outputs = _run(tmp_path, spacing=(1.0, 1.0, 0.0251 * 3), hu=400)
    assert 0 < result["cacs"] < 0.5
    assert outputs[0].value == result["cacs"]
    assert outputs[1].value == 1
    assert result["lesion_count"] == 1
    assert outputs[5].value == SCORING_VERSION


@pytest.mark.parametrize("mismatch", [
    ("Origin", (1, 0, 0)),
    ("Spacing", (2, 1, 3)),
    ("Direction", (-1, 0, 0, 0, 1, 0, 0, 0, 1)),
])
def test_reject_geometry_mismatch(tmp_path, mismatch):
    with pytest.raises(ValueError, match="geometry mismatch"):
        _run(tmp_path, mismatch=mismatch)


@pytest.mark.parametrize("workflow_name", ["default.yml", "nifti_pid.yml", "dicom_dicom_plastimatch.yml", "dicom_extended.yml"])
@pytest.mark.parametrize("n_lesions", [0, 1, 2])
def test_decorated_module_registers_outputs_and_report(tmp_path, monkeypatch, workflow_name, n_lesions):
    from mhubio.core import Config, DataType, InstanceData
    from mhubio.modules.exporter.ReportExporter import ReportExporter
    import yaml

    monkeypatch.setenv("MLOG_STOP_ON_ERROR", "YES")
    workflow = Path(__file__).parents[1] / "config" / workflow_name
    settings = yaml.safe_load(workflow.read_text())
    settings.pop("segdb", None)  # This test exercises scoring/export rather than SEG generation.
    settings["general"]["data_base_dir"] = str(tmp_path)
    config = Config(config=settings, args=False)
    instance = Instance("case")
    config.data.addInstance(instance)
    instance.attr["sid"] = "1.2.3"
    if workflow_name == "nifti_pid.yml":
        instance.attr["pid"] = "TEST"
    image = InstanceData("ct.nii.gz", DataType.fromString("nifti:mod=ct"), instance=instance)
    mask = InstanceData("cac.nrrd", DataType.fromString("nrrd:mod=seg:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION"), instance=instance)
    tmp_path.joinpath("case").mkdir()
    ct_array = np.full((2, 5, 5), 400, dtype=np.int16)
    mask_array = np.zeros_like(ct_array, dtype=np.uint8)
    if n_lesions >= 1:
        mask_array[:, :2, :2] = 1
    if n_lesions == 2:
        mask_array[0, 3:5, 3:5] = 1
        ct_array[0, 3:5, 3:5] = 300
    ct = sitk.GetImageFromArray(ct_array)
    ct.SetSpacing((0.5, 0.5, 3))
    cac = sitk.GetImageFromArray(mask_array)
    cac.CopyInformation(ct)
    sitk.WriteImage(ct, image.abspath)
    sitk.WriteImage(cac, mask.abspath)
    image.confirm()
    mask.confirm()
    if workflow_name != "nifti_pid.yml":
        # An unreadable DICOM entry must not affect the NIfTI-only calculator.
        # DICOM consumers such as the SR adapter are tested separately.
        # DicomImporter stores PatientID on input metadata, not instance.attr.
        source = InstanceData("unread-dicom", DataType.fromString("dicom:mod=ct:pid=TEST"), instance=instance)
        source.confirm()
    CalciumAnalysisCalculator(config).task()
    assert len(instance.data.filter("json:mod=calcium_measurements")) == 0
    loaded_ct = sitk.ReadImage(image.abspath)
    result = analyze_calcium(sitk.GetArrayFromImage(loaded_ct), mask_array, loaded_ct.GetSpacing())
    expected_score = [0.0, 8.0, 11.0][n_lesions]
    assert result["cacs"] == expected_score
    assert instance.outputData.filter("cacs").first().value == expected_score
    from models.deepcac2.utils.TID1500SRGenerator import measurements_from_outputs
    sr_values = measurements_from_outputs(instance)
    for key in ("cacs", "calcium_volume_mm3", "segmentation_volume_mm3", "lesion_count"):
        assert sr_values[key] == result[key]
    assert sr_values["lesions"] == [
        {key: lesion[key] for key in ("id", "cacs", "volume_mm3", "max_hu")}
        for lesion in result["lesions"]
    ]
    fields = [
        ("lesion_id", "id", "Lesion IDs", "1"),
        ("lesion_volume", "volume_mm3", "Individual Lesion Volumes (mm3)", "mm3"),
        ("lesion_score", "cacs", "Individual Lesion CAC Scores", "1"),
        ("lesion_max_hu", "max_hu", "Individual Lesion Maximum HU", "[hnsf'U]"),
        ("lesion_eligible_voxels", "eligible_voxels", "Individual Lesion Eligible Voxel Counts", "1"),
        ("lesion_region_count", "n_regions_2d", "Individual Lesion 2D Region Counts", "1"),
    ]
    for name, key, _, unit in fields:
        outputs = list(instance.outputData.filter(name))
        assert len(outputs) == n_lesions
        assert [o.value for o in outputs[:5]] == [l[key] for l in result["lesions"]]
        assert [o.meta["id"] for o in outputs] == [str(l["id"]) for l in result["lesions"]]
        assert all(o.meta["unit"] == unit for o in outputs)
        assert all(o.meta["roi"] == "CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION" for o in outputs)
    if n_lesions:
        assert instance.outputData.filter("lesion_volume:id=1").first().value == result["lesions"][0]["volume_mm3"]
    ReportExporter(config).task()
    report = instance.data.filter("json:mod=report").first()
    values = json.loads(Path(report.abspath).read_text())
    assert values["PatientID"] == "TEST"
    assert values["Scoring Version"] == SCORING_VERSION
    assert values["Lesion Connectivity"] == 26
    assert values["Predicted CAC Score"] == expected_score
    assert values["Predicted Risk Group"] == (1 if n_lesions else 0)
    assert values["Calcium Volume (mm3)"] == [0.0, 6.0, 9.0][n_lesions]
    for _, key, label, _ in fields:
        assert values[label] == [l[key] for l in result["lesions"]]
    if n_lesions:
        assert sum(values["Individual Lesion CAC Scores"]) == values["Predicted CAC Score"]
        assert sum(values["Individual Lesion Volumes (mm3)"]) == values["Calcium Volume (mm3)"]
    global_steps = [step for step in settings["execute"] if isinstance(step, dict) and step.get("globalreport")]
    assert len(global_steps) == 1
    ReportExporter(config, local_config={"globalreport": True, "meta": {"scope": "global"}}).task()
    combined = config.data.globalInstance.data.filter("json:mod=report").first()
    assert combined.confirmed
    assert json.loads(Path(combined.abspath).read_text()) == [values]

    if workflow_name != "nifti_pid.yml":
        assert len(values) == 15
        from mhubio.modules.organizer.DataOrganizer import DataOrganizer
        # MHub's target registry is class-scoped. Each parametrized workflow
        # represents a fresh process, so do not carry targets from earlier runs.
        monkeypatch.setattr(DataOrganizer, "_targets", {})
        for filename, dtype in [("seg.dcm", "dicomseg:mod=seg"), ("sr.dcm", "dicom:mod=sr:tid=1500")]:
            data = InstanceData(filename, DataType.fromString(dtype), instance=instance)
            Path(data.abspath).write_bytes(filename.encode())
            data.confirm()
        DataOrganizer(config).task()
        exported = {str(path.relative_to(tmp_path / "output_data")) for path in (tmp_path / "output_data").rglob("*") if path.is_file()}
        expected = {"DeepCAC.report.json", "1.2.3/DeepCAC2.seg.dcm", "1.2.3/DeepCAC2.tid1500.dcm", "1.2.3/DeepCAC.report.json"}
        if workflow_name == "dicom_extended.yml":
            expected.add("1.2.3/nrrd/cac.seg.nrrd")
        assert exported == expected


def test_configured_connectivity_is_published(tmp_path):
    result, outputs = _run(tmp_path, hu=400, connectivity=6)
    assert result["lesion_connectivity"] == outputs[6].value == 6


@pytest.mark.parametrize("workflow_name", ["default.yml", "dicom_dicom_plastimatch.yml", "dicom_extended.yml"])
@pytest.mark.parametrize("patient_id", ["100002", None])
def test_imported_patient_id_is_exported_without_instance_attribute(tmp_path, monkeypatch, workflow_name, patient_id):
    """Exercise the actual importer metadata path, including absent PatientID."""
    import yaml
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    from mhubio.core import Config
    from mhubio.modules.importer.DicomImporter import DicomImporter
    from mhubio.modules.exporter.ReportExporter import ReportExporter
    from models.deepcac2.tests.test_dicom_sr import _evidence

    monkeypatch.setenv("MLOG_STOP_ON_ERROR", "YES")
    settings = yaml.safe_load((Path(__file__).parents[1] / "config" / workflow_name).read_text())
    settings.pop("segdb", None)
    settings["general"]["data_base_dir"] = str(tmp_path)
    patient_field = next(field for field in settings["modules"]["ReportExporter"]["includes"] if field["label"] == "PatientID")
    settings["modules"]["ReportExporter"]["includes"] = [patient_field]
    config = Config(config=settings, args=False)
    source, _ = _evidence()
    if patient_id is None:
        del source.PatientID
    else:
        source.PatientID = patient_id
    source.file_meta = FileMetaDataset()
    source.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    source.is_little_endian = True
    source.is_implicit_VR = False
    sorted_dir = tmp_path / "sorted_data"
    dicom_dir = sorted_dir / str(source.SeriesInstanceUID) / "dicom"
    dicom_dir.mkdir(parents=True)
    source.save_as(str(dicom_dir / "ct.dcm"), write_like_original=False)
    DicomImporter(config).importSorted(str(sorted_dir))
    instance = config.data.instances[0]
    assert "pid" not in instance.attr
    assert ReportExporter(config).generateInstanceReport(instance) == {"PatientID": patient_id or ""}
