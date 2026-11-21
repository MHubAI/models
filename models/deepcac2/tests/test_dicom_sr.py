"""Focused tests for DICOM SR construction."""

import tempfile
import unittest

import highdicom as hd
from pydicom.dataset import Dataset
from pydicom.uid import CTImageStorage, SegmentationStorage, generate_uid

from models.deepcac2.src.dicom_sr import build_tid1500_report
from models.deepcac2.src.scoring import SCORING_VERSION, classify_risk


def _evidence_dataset(study_uid, series_uid, sop_class_uid, modality):
    dataset = Dataset()
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = generate_uid()
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.PatientID = "TEST"
    dataset.PatientName = "Test^Patient"
    dataset.PatientBirthDate = "19700101"
    dataset.PatientSex = "O"
    dataset.AccessionNumber = ""
    dataset.StudyID = "1"
    dataset.StudyDate = "20260904"
    dataset.StudyTime = "120000"
    dataset.Modality = modality
    return dataset


def _evidence():
    study_uid = generate_uid()
    image_series_uid = generate_uid()
    source = _evidence_dataset(
        study_uid, image_series_uid, CTImageStorage, "CT"
    )
    segmentation = _evidence_dataset(
        study_uid, generate_uid(), SegmentationStorage, "SEG"
    )
    segment = Dataset()
    segment.SegmentNumber = 7
    property_type = Dataset()
    property_type.CodeValue = "18115005"
    property_type.CodingSchemeDesignator = "SCT"
    property_type.CodeMeaning = "Pathologic calcification"
    segment.SegmentedPropertyTypeCodeSequence = [property_type]
    segmentation.SegmentSequence = [segment]
    return source, segmentation


def _heart_only_evidence():
    """Build the SEG shape produced when DsegConverter omits empty CAC."""
    source, segmentation = _evidence()
    heart = Dataset()
    heart.SegmentNumber = 1
    property_type = Dataset()
    property_type.CodeValue = "80891009"
    property_type.CodingSchemeDesignator = "SCT"
    property_type.CodeMeaning = "Heart structure"
    heart.SegmentedPropertyTypeCodeSequence = [property_type]
    segmentation.SegmentSequence = [heart]
    return source, segmentation


def _find_descendant_by_code(dataset, code_value):
    matches = []
    for child in getattr(dataset, "ContentSequence", []):
        names = getattr(child, "ConceptNameCodeSequence", [])
        if names and str(names[0].CodeValue) == code_value:
            matches.append(child)
        matches.extend(_find_descendant_by_code(child, code_value))
    return matches


def _measurements(score=115, volumes=(4.0, 2.0), segmentation_volume=None):
    return dict(cacs=score, risk_category=classify_risk(score), scoring_version=SCORING_VERSION,
                lesion_connectivity=26, calcium_volume_mm3=sum(volumes),
                segmentation_volume_mm3=sum(volumes) if segmentation_volume is None else segmentation_volume,
                lesion_count=len(volumes), lesions=[dict(id=i, cacs=score / len(volumes),
                volume_mm3=v, max_hu=400) for i, v in enumerate(volumes, 1)])


class StructuredReportTests(unittest.TestCase):
    def test_tid1500_report_references_segmentation(self):
        source, segmentation = _evidence()
        report = build_tid1500_report(
            [source], segmentation, _measurements()
        )

        self.assertEqual(report.ContentTemplateSequence[0].TemplateIdentifier, "1500")
        procedure = _find_descendant_by_code(report, "121058")[0]
        self.assertEqual(
            str(procedure.ConceptCodeSequence[0].CodeValue), "363679005"
        )
        with tempfile.NamedTemporaryFile(suffix=".dcm") as output:
            report.save_as(output.name, write_like_original=False)
            parsed = hd.sr.srread(output.name)
        groups = parsed.content.get_volumetric_roi_measurement_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].get_measurements()), 1)
        referenced_segments = _find_descendant_by_code(report, "121191")
        self.assertEqual(len(referenced_segments), 1)
        reference = referenced_segments[0].ReferencedSOPSequence[0]
        self.assertEqual(reference.ReferencedSOPInstanceUID, segmentation.SOPInstanceUID)
        self.assertEqual(reference.ReferencedSegmentNumber, 7)

    def test_tid1500_supports_cac_negative_cases(self):
        source, segmentation = _heart_only_evidence()
        tid1500 = build_tid1500_report(
            [source], segmentation, _measurements(0, ())
        )
        self.assertEqual(len(_find_descendant_by_code(tid1500, "246206008")), 1)
        self.assertEqual(len(_find_descendant_by_code(tid1500, "121191")), 0)
        evidence_uids = {
            str(item.ReferencedSOPInstanceUID)
            for sequence in [*getattr(tid1500, "PertinentOtherEvidenceSequence", []),
                             *getattr(tid1500, "CurrentRequestedProcedureEvidenceSequence", [])]
            for series in sequence.ReferencedSeriesSequence
            for item in series.ReferencedSOPSequence
        }
        self.assertNotIn(str(segmentation.SOPInstanceUID), evidence_uids)

    def test_nonempty_segmentation_without_eligible_calcium(self):
        source, segmentation = _evidence()
        report = build_tid1500_report(
            [source], segmentation, _measurements(0, (), segmentation_volume=12.0)
        )
        self.assertEqual(len(_find_descendant_by_code(report, "121191")), 1)
        volume = _find_descendant_by_code(report, "118565006")[0]
        self.assertEqual(float(volume.MeasuredValueSequence[0].NumericValue), 12.0)
        calcium = _find_descendant_by_code(report, "122660")[0]
        self.assertEqual(float(calcium.MeasuredValueSequence[0].NumericValue), 0.0)

    def test_raw_positive_segmentation_requires_cac_segment(self):
        source, segmentation = _heart_only_evidence()
        with self.assertRaisesRegex(ValueError, "contains no.*segment"):
            build_tid1500_report(
                [source], segmentation, _measurements(0, (), segmentation_volume=12.0)
            )

    def test_tid1500_rejects_positive_result_without_cac_segment(self):
        source, segmentation = _heart_only_evidence()
        with self.assertRaisesRegex(ValueError, "contains no.*segment"):
            build_tid1500_report(
                [source], segmentation, _measurements(1, (2.0,))
            )


if __name__ == "__main__":
    unittest.main()


def test_negative_tracking_ids_distinguish_series_in_one_study():
    from copy import deepcopy
    source, segmentation = _heart_only_evidence()
    other_series = deepcopy(source)
    other_series.SeriesInstanceUID = generate_uid()
    other_series.SOPInstanceUID = generate_uid()

    def tracking_uid(image):
        report = build_tid1500_report([image], segmentation, _measurements(0, ()))
        return str(_find_descendant_by_code(report, "112040")[0].UID)

    assert tracking_uid(source) == tracking_uid(source)
    assert tracking_uid(source) != tracking_uid(other_series)


def test_sr_records_actual_scoring_version_connectivity_and_risk_definition():
    source, segmentation = _evidence()
    values = _measurements()
    values.update(scoring_version="test-version", lesion_connectivity=18)
    report = build_tid1500_report([source], segmentation, values)
    parameters = [str(item.TextValue) for item in _find_descendant_by_code(report, "111002")]
    assert any("3D lesion connectivity: 18" in item for item in parameters)
    assert any("0 = 0; 1 = (0,100]; 2 = (100,300]; 3 = >300" in item for item in parameters)
    versions = _find_descendant_by_code(report, "111003")
    assert all(str(item.TextValue) == "test-version" for item in versions)
    assert report.SoftwareVersions == "test-version"
    assert report.ManufacturerModelName == "DeepCAC2"


def test_wrong_coding_scheme_cannot_be_selected_as_cac():
    import pytest
    source, segmentation = _evidence()
    segmentation.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator = "99OTHER"
    with pytest.raises(ValueError, match="contains no.*segment"):
        build_tid1500_report([source], segmentation, _measurements())


def test_ambiguous_cac_segments_are_rejected():
    from copy import deepcopy
    import pytest
    source, segmentation = _evidence()
    extra = deepcopy(segmentation.SegmentSequence[0])
    extra.SegmentNumber = 8
    segmentation.SegmentSequence.append(extra)
    with pytest.raises(ValueError, match="multiple pathologic-calcification"):
        build_tid1500_report([source], segmentation, _measurements())
