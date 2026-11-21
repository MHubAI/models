"""SR adapter tests using native MHub measurements and no JSON dependency."""
from pathlib import Path

import pytest
import highdicom as hd
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

pytest.importorskip("mhubio")
from mhubio.core import Config, DataType, Instance, InstanceData, Meta, OutputDataCollection
from models.deepcac2.utils.CalciumAnalysisCalculator import (
    Cacs, RiskCategory, ScoringVersion, LesionConnectivity, CalciumVolume, SegmentationVolume, LesionCount, LesionId,
    LesionVolume, LesionScore, LesionMaxHU,
)
from models.deepcac2.src.scoring import SCORING_VERSION
from models.deepcac2.utils.TID1500SRGenerator import (
    TID1500SRGenerator, measurements_from_outputs,
)
from models.deepcac2.tests.test_dicom_sr import (
    _evidence, _heart_only_evidence, _find_descendant_by_code,
)


def _add(instance, schema, value, lesion_id=None):
    output = schema()
    output.value = value
    if lesion_id is not None:
        output.meta = output.meta + Meta(id=str(lesion_id))
    instance.outputData.add(output)
    return output


def _registered(positive=True, path="case"):
    instance = Instance(path)
    for schema, value in [(ScoringVersion, SCORING_VERSION), (LesionConnectivity, 26), (Cacs, 7.0 if positive else 0.0),
                          (CalciumVolume, 5.0 if positive else 0.0),
                          (SegmentationVolume, 6.0 if positive else 0.0),
                          (LesionCount, 2 if positive else 0),
                          (RiskCategory, 1 if positive else 0)]:
        _add(instance, schema, value)
    if positive:
        # Interleave and reverse the output order: pairing must use metadata IDs.
        for schema, values in [
            (LesionVolume, [(7, 3.0), (2, 2.0)]),
            (LesionId, [(2, 2), (7, 7)]),
            (LesionMaxHU, [(2, 250.0), (7, 450.0)]),
            (LesionScore, [(7, 4.25), (2, 2.75)]),
        ]:
            for lesion_id, value in values:
                _add(instance, schema, value, lesion_id)
    return instance


def test_join_by_id_instead_of_output_order():
    result = measurements_from_outputs(_registered())
    assert result["lesions"] == [
        dict(id=2, volume_mm3=2.0, cacs=2.75, max_hu=250.0),
        dict(id=7, volume_mm3=3.0, cacs=4.25, max_hu=450.0),
    ]
    assert result["lesion_count"] == 2


@pytest.mark.parametrize("cases", [(False,), (True,), (True, True, False)])
def test_decorated_sr_adapter_without_measurement_json(tmp_path, monkeypatch, cases):
    monkeypatch.setenv("MLOG_STOP_ON_ERROR", "YES")
    config = Config(config={"general": {"data_base_dir": str(tmp_path)}}, args=False)
    expected = []
    for index, positive in enumerate(cases):
        path = f"case-{index}"
        instance = _registered(positive, path=path)
        # Give the positive cases different scores/volumes but the same lesion
        # IDs, so cross-instance mixing cannot pass by producing identical SRs.
        factor = index + 1
        for name in ("cacs", "calcium_volume", "segmentation_volume", "lesion_score", "lesion_volume"):
            for output in instance.outputData.filter(name):
                output.value *= factor
        config.data.addInstance(instance)
        directory = tmp_path / path / "dicom"
        directory.mkdir(parents=True)
        source, seg = _evidence() if positive else _heart_only_evidence()
        for dataset, target in [(source, directory / "ct.dcm"), (seg, tmp_path / path / "seg.dcm")]:
            dataset.file_meta = FileMetaDataset()
            dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset.is_little_endian = True
            dataset.is_implicit_VR = False
            dataset.save_as(str(target), write_like_original=False)
        for filename, dtype in [("dicom", "dicom:mod=ct"), ("seg.dcm", "dicomseg:mod=seg")]:
            data = InstanceData(filename, DataType.fromString(dtype), instance=instance)
            data.confirm()
        assert len(instance.data.filter("json:mod=calcium_measurements")) == 0
        expected.append((instance, source, seg, positive, factor))

    # One module call lets MHub iterate all instances, rather than invoking the
    # underlying Python function separately in the test.
    TID1500SRGenerator(config).task()
    paths = set()
    for instance, source, seg, positive, factor in expected:
        output = instance.data.filter("dicom:mod=sr:tid=1500").first()
        assert output.confirmed and Path(output.abspath).is_file()
        paths.add(output.abspath)
        parsed = hd.sr.srread(output.abspath)
        assert parsed.StudyInstanceUID == source.StudyInstanceUID
        assert parsed.ContentTemplateSequence[0].TemplateIdentifier == "1500"
        assert len(parsed.content.get_volumetric_roi_measurement_groups()) == int(positive)
        scores = _find_descendant_by_code(parsed, "450360000")
        actual = [float(item.MeasuredValueSequence[0].NumericValue) for item in scores]
        assert actual == ([7.0 * factor, 2.75 * factor, 4.25 * factor] if positive else [0.0])
        volumes = _find_descendant_by_code(parsed, "122660")
        assert float(volumes[0].MeasuredValueSequence[0].NumericValue) == (5.0 * factor if positive else 0.0)
        segments = _find_descendant_by_code(parsed, "121191")
        if positive:
            reference = segments[0].ReferencedSOPSequence[0]
            assert reference.ReferencedSOPInstanceUID == seg.SOPInstanceUID
        else:
            assert segments == []
    assert len(paths) == len(cases)


@pytest.mark.parametrize("name", ["cacs", "calcium_volume", "segmentation_volume", "lesion_count", "risk_category", "scoring_version", "lesion_connectivity"])
def test_duplicate_summary_is_rejected(name):
    instance = _registered()
    original = instance.outputData.filter(name).first()
    _add(instance, type(original), original.value)
    with pytest.raises(ValueError, match="Expected exactly one"):
        measurements_from_outputs(instance)


@pytest.mark.parametrize("field", ["lesion_id", "lesion_volume", "lesion_score", "lesion_max_hu"])
def test_duplicate_lesion_measurement_is_rejected(field):
    instance = _registered()
    original = instance.outputData.filter(field).first()
    _add(instance, type(original), original.value, original.meta["id"])
    with pytest.raises(ValueError, match="Duplicate"):
        measurements_from_outputs(instance)


@pytest.mark.parametrize("field", ["lesion_volume", "lesion_score", "lesion_max_hu"])
def test_mismatched_lesion_ids_are_rejected(field):
    instance = _registered()
    instance.outputData.filter(field).first().meta = Meta(id="99")
    with pytest.raises(ValueError, match="IDs do not match"):
        measurements_from_outputs(instance)


def test_missing_lesion_metadata_is_rejected():
    instance = _registered()
    instance.outputData.filter("lesion_volume").first().meta = Meta()
    with pytest.raises(ValueError, match="meta.id"):
        measurements_from_outputs(instance)


@pytest.mark.parametrize("field,value,error", [
    ("lesion_count", 3, "lesion_count"),
    ("risk_category", 2, "Risk category"),
    ("cacs", 8.0, "sum disagrees"),
    ("calcium_volume", 4.0, "sum disagrees"),
    ("segmentation_volume", 1.0, "exceeds"),
    ("lesion_max_hu", float("nan"), "finite number"),
])
def test_inconsistent_measurements_are_rejected(field, value, error):
    instance = _registered()
    instance.outputData.filter(field).first().value = value
    with pytest.raises(ValueError, match=error):
        measurements_from_outputs(instance)


@pytest.mark.parametrize("field", ["cacs", "calcium_volume", "segmentation_volume", "lesion_count", "risk_category",
                                   "lesion_id", "lesion_volume", "lesion_score", "lesion_max_hu"])
def test_missing_required_output_is_rejected(field):
    instance = _registered()
    missing = instance.outputData.filter(field).first()
    instance.outputData = OutputDataCollection([o for o in instance.outputData if o is not missing])
    with pytest.raises(ValueError):
        measurements_from_outputs(instance)


def test_totals_accept_roundoff_without_changing_values():
    instance = _registered()
    original = instance.outputData.filter("cacs").first()
    original.value = 7.0 + 1e-12
    assert measurements_from_outputs(instance)["cacs"] == original.value


def test_nonempty_raw_segmentation_with_zero_eligible_calcium():
    instance = _registered(positive=False)
    instance.outputData.filter("segmentation_volume").first().value = 6.0
    result = measurements_from_outputs(instance)
    assert result["lesions"] == []
    assert result["cacs"] == result["calcium_volume_mm3"] == 0
    assert result["segmentation_volume_mm3"] == 6.0


def test_sr_requires_explicit_seg_input(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOG_STOP_ON_ERROR", "YES")
    config = Config(config={"general": {"data_base_dir": str(tmp_path)}}, args=False)
    instance = _registered(positive=False)
    config.data.addInstance(instance)
    source = InstanceData("dicom", DataType.fromString("dicom:mod=ct"), instance=instance)
    source.confirm()
    from mhubio.core.Error import MHubMissingDataError
    with pytest.raises(MHubMissingDataError):
        TID1500SRGenerator(config).task()
    assert len(instance.data.filter("dicom:mod=sr")) == 0
