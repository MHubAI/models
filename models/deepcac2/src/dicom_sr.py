"""DICOM SR builders for DeepCAC2 plaque and calcium measurements."""

from pathlib import Path
from typing import List, Optional, Sequence

import highdicom as hd
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from .scoring import HU_THRESHOLD, MIN_AREA_MM2, REFERENCE_THICKNESS_MM


MODEL_NAME = "DeepCAC2"
MANUFACTURER = "MHub.ai"
DEVICE_UID = generate_uid(entropy_srcs=[MODEL_NAME, MANUFACTURER])


def _code(value: str, scheme: str, meaning: str) -> hd.sr.CodedConcept:
    return hd.sr.CodedConcept(value, scheme, meaning)


def DCM(value: str, meaning: str) -> hd.sr.CodedConcept:
    return _code(value, "DCM", meaning)


def SCT(value: str, meaning: str) -> hd.sr.CodedConcept:
    return _code(value, "SCT", meaning)


def UCUM(value: str, meaning: str) -> hd.sr.CodedConcept:
    return _code(value, "UCUM", meaning)


IMAGING_PROCEDURE = SCT("363679005", "Imaging procedure")
CORONARY_CALCIFICATION = SCT("445512009", "Calcification of coronary artery")
PATHOLOGIC_CALCIFICATION = SCT("18115005", "Pathologic calcification")
CORONARY_ARTERY = SCT("41801008", "Coronary artery structure")
MORPHOLOGICALLY_ABNORMAL = SCT(
    "49755003", "Morphologically abnormal structure"
)
VOLUME = SCT("118565006", "Volume")
CALCIUM_VOLUME = DCM("122660", "Calcium Volume")
NUMBER_OF_LESIONS = SCT("246206008", "Number of Lesions")
CALCIUM_SCORE = SCT("450360000", "Coronary artery calcium score")
AGATSTON_METHOD = DCM("112055", "Agatston Scoring Method")
MM3 = UCUM("mm3", "mm3")
LESIONS = UCUM("{lesions}", "lesions")
NO_UNITS = UCUM("1", "no units")
HU = UCUM("[hnsf'U]", "Hounsfield unit")
RISK = _code("RISK", "99DEEPCAC2", "Calcium score risk category")
MAX_HU = _code("MAXHU", "99DEEPCAC2", "Maximum attenuation")


def load_dicom_series(directory: str) -> List[Dataset]:
    """Load image headers from an MHub DICOM series directory."""
    datasets: List[Dataset] = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if hasattr(dataset, "SOPClassUID") and hasattr(dataset, "SOPInstanceUID"):
            datasets.append(dataset)
    if not datasets:
        raise ValueError(f"No readable DICOM instances found in {directory}.")
    return datasets


def _observer_context() -> hd.sr.ObservationContext:
    device = hd.sr.DeviceObserverIdentifyingAttributes(
        uid=DEVICE_UID,
        name=MODEL_NAME,
        manufacturer_name=MANUFACTURER,
        model_name=MODEL_NAME,
    )
    return hd.sr.ObservationContext(
        observer_device_context=hd.sr.ObserverContext(
            observer_type=DCM("121007", "Device"),
            observer_identifying_attributes=device,
        )
    )


def _tracking_identifier(source_uid: str, identifier: str, scoring_version: str):
    """Identify a measurement group within its SEG or source CT series."""
    return hd.sr.TrackingIdentifier(
        uid=generate_uid(entropy_srcs=[MODEL_NAME, scoring_version, str(source_uid), identifier]),
        identifier=identifier,
    )


def _algorithm(scoring_version: str, connectivity: int) -> hd.sr.AlgorithmIdentification:
    """Describe the scoring method and configured lesion grouping."""
    return hd.sr.AlgorithmIdentification(
        name=f"{MODEL_NAME} calcium analysis",
        version=scoring_version,
        parameters=[
            f"HU >= {HU_THRESHOLD}; 2D 4-connectivity; area >= {MIN_AREA_MM2:g} mm2; "
            f"slice spacing / {REFERENCE_THICKNESS_MM:g} mm",
            f"3D lesion connectivity: {connectivity}",
            "Risk categories from unrounded CACS: 0 = 0; 1 = (0,100]; 2 = (100,300]; 3 = >300",
        ],
    )


def _cac_segment_number(segmentation: Dataset) -> Optional[int]:
    """Select the CAC segment by its full coded identity, never by display text."""
    matches = []
    for segment in getattr(segmentation, "SegmentSequence", []):
        codes = getattr(segment, "SegmentedPropertyTypeCodeSequence", [])
        if any(
            str(code.CodeValue) == PATHOLOGIC_CALCIFICATION.value
            and str(code.CodingSchemeDesignator) == PATHOLOGIC_CALCIFICATION.scheme_designator
            for code in codes
        ):
            matches.append(int(segment.SegmentNumber))
    if len(matches) > 1:
        raise ValueError("DICOM SEG contains multiple pathologic-calcification segments.")
    return matches[0] if matches else None


def build_tid1500_report(
    source_images: Sequence[Dataset],
    segmentation: Optional[Dataset],
    measurements: dict,
    series_number: int = 1500,
) -> hd.sr.Comprehensive3DSR:
    """Build global, raw-segmentation and non-spatial lesion measurement groups."""
    if not source_images:
        raise ValueError("At least one source image is required.")
    cac_segment_number = (
        _cac_segment_number(segmentation) if segmentation is not None else None
    )
    # A raw-positive but score-negative mask still requires CAC SEG evidence.
    measurements_indicate_cac = any(measurements[key] != 0 for key in (
        "segmentation_volume_mm3", "calcium_volume_mm3", "lesion_count", "cacs",
    ))
    if cac_segment_number is None and measurements_indicate_cac:
        raise ValueError(
            "CAC measurements are non-zero, but the DICOM SEG contains no "
            "pathologic-calcification segment."
        )

    scoring_version = measurements["scoring_version"]
    # Negative series within one study must receive distinct tracking identities.
    tracking_source_uid = (
        segmentation.SOPInstanceUID if cac_segment_number is not None
        else source_images[0].SeriesInstanceUID
    )
    common = dict(
        finding_type=CORONARY_CALCIFICATION,
        finding_category=MORPHOLOGICALLY_ABNORMAL,
        finding_sites=[hd.sr.FindingSite(CORONARY_ARTERY)],
        algorithm_id=_algorithm(scoring_version, measurements["lesion_connectivity"]),
    )

    # Global measurements describe eligible calcium, not every voxel in the mask.
    groups = [hd.sr.MeasurementsAndQualitativeEvaluations(
        tracking_identifier=_tracking_identifier(
            tracking_source_uid, "DeepCAC2 calcium measurements", scoring_version),
        source_images=[
            hd.sr.SourceImageForMeasurementGroup(
                referenced_sop_class_uid=image.SOPClassUID,
                referenced_sop_instance_uid=image.SOPInstanceUID,
            ) for image in source_images
        ],
        measurements=[
            hd.sr.Measurement(name=CALCIUM_SCORE, value=measurements["cacs"],
                              unit=NO_UNITS, method=AGATSTON_METHOD),
            hd.sr.Measurement(name=CALCIUM_VOLUME, value=measurements["calcium_volume_mm3"], unit=MM3),
            hd.sr.Measurement(name=NUMBER_OF_LESIONS, value=measurements["lesion_count"], unit=LESIONS),
            hd.sr.Measurement(name=RISK, value=measurements["risk_category"], unit=NO_UNITS),
        ],
        **common,
    )]
    evidence = list(source_images)

    # Only raw-mask volume references the complete CAC segment as a spatial ROI.
    if cac_segment_number is not None:
        groups.append(hd.sr.VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=_tracking_identifier(
                tracking_source_uid, "DeepCAC2 CAC segmentation", scoring_version),
            referenced_segment=hd.sr.ReferencedSegment(
                sop_class_uid=segmentation.SOPClassUID,
                sop_instance_uid=segmentation.SOPInstanceUID,
                segment_number=cac_segment_number,
                source_series=hd.sr.SourceSeriesForSegmentation(source_images[0].SeriesInstanceUID),
            ),
            measurements=[hd.sr.Measurement(
                name=VOLUME, value=measurements["segmentation_volume_mm3"], unit=MM3)],
            **common,
        ))
        evidence.append(segmentation)

    # Individual lesions have IDs and measurements, but no separate SEG segments.
    for lesion in measurements["lesions"]:
        groups.append(hd.sr.MeasurementsAndQualitativeEvaluations(
            tracking_identifier=_tracking_identifier(
                tracking_source_uid, f"DeepCAC2 lesion {lesion['id']}", scoring_version),
            measurements=[
                hd.sr.Measurement(name=CALCIUM_SCORE, value=lesion["cacs"],
                                  unit=NO_UNITS, method=AGATSTON_METHOD),
                hd.sr.Measurement(name=VOLUME, value=lesion["volume_mm3"], unit=MM3),
                hd.sr.Measurement(name=MAX_HU, value=lesion["max_hu"], unit=HU),
            ],
            **common,
        ))

    report = hd.sr.MeasurementReport(
        observation_context=_observer_context(),
        # Generic procedure fallback when no coded performed procedure is supplied.
        procedure_reported=IMAGING_PROCEDURE,
        imaging_measurements=groups,
        title=DCM("126000", "Imaging Measurement Report"),
    )
    return hd.sr.Comprehensive3DSR(
        evidence=evidence,
        content=report,
        series_instance_uid=generate_uid(),
        series_number=series_number,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        manufacturer_model_name=MODEL_NAME,
        software_versions=scoring_version,
        is_complete=True,
        is_final=False,
        series_description="DeepCAC2 TID 1500 Measurement Report",
    )
