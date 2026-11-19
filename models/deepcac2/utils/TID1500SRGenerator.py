"""MHub module that exports calcium measurements as a TID 1500 DICOM SR."""

from math import isclose, isfinite
from numbers import Integral, Real

import pydicom

from mhubio.core import IO, ClassOutput, Instance, InstanceData, Module, ValueOutput

from ..src.dicom_sr import build_tid1500_report, load_dicom_series
from ..src.scoring import classify_risk


# Aggregating lesion contributions changes the order of floating-point additions.
# Allow roundoff without rounding values or altering the score used for risk.
TOTAL_REL_TOL = 1e-9
TOTAL_ABS_TOL = 1e-9


def _totals_match(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=TOTAL_REL_TOL, abs_tol=TOTAL_ABS_TOL)


def _number(output, name, integer=False, output_type=ValueOutput):
    """Read a numeric output without silently truncating counts or accepting NaN."""
    if not isinstance(output, output_type):
        raise ValueError(f"{name} must be a {output_type.__name__}.")
    value = output.value
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if integer and not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value) if integer else float(value)


def _single_output(instance, name):
    """Require one summary output in this instance, never choose an arbitrary match."""
    outputs = list(instance.outputData.filter(name))
    if len(outputs) != 1:
        raise ValueError(f"Expected exactly one {name} output, found {len(outputs)}.")
    return outputs[0]


def _scalar(instance, name, integer=False, output_type=ValueOutput):
    return _number(_single_output(instance, name), name, integer, output_type)


def _by_lesion_id(instance, name, integer=False):
    """Index one measurement per lesion; output collection order is irrelevant."""
    values = {}
    for output in instance.outputData.filter(name):
        raw_id = None
        if output.meta is not None and "id" in output.meta:
            raw_id = output.meta["id"]
        try:
            lesion_id = int(str(raw_id))
        except (TypeError, ValueError):
            raise ValueError(f"{name} requires a positive integer meta.id.") from None
        if lesion_id <= 0:
            raise ValueError(f"{name} requires a positive integer meta.id.")
        if lesion_id in values:
            raise ValueError(f"Duplicate {name} output for lesion {lesion_id}.")
        values[lesion_id] = _number(output, name, integer)
    return values


def measurements_from_outputs(instance: Instance) -> dict:
    """Assemble SR inputs and validate their consistency without rescoring calcium.

    These checks validate the producer/consumer contract. HU eligibility, area
    cutoffs, connectivity and normalization remain the scoring module's job.
    """
    # Each MHub instance owns its outputData collection. Repeated lesion IDs in
    # other instances cannot collide with the measurements selected here.
    result = {
        "cacs": _scalar(instance, "cacs"),
        "calcium_volume_mm3": _scalar(instance, "calcium_volume"),
        "segmentation_volume_mm3": _scalar(instance, "segmentation_volume"),
        "lesion_count": _scalar(instance, "lesion_count", integer=True),
    }
    if any(value < 0 for value in result.values()):
        raise ValueError("Calcium summary measurements must be non-negative.")
    # Confirm the published category uses the unrounded score; do not replace it.
    risk_category = _scalar(instance, "risk_category", integer=True, output_type=ClassOutput)
    if risk_category != classify_risk(result["cacs"]):
        raise ValueError("Risk category disagrees with the unrounded CAC score.")

    result["risk_category"] = risk_category
    result["scoring_version"] = _single_output(instance, "scoring_version").value
    result["lesion_connectivity"] = _scalar(instance, "lesion_connectivity", integer=True)

    # The explicit ID outputs define the complete lesion set. Missing fields
    # must raise an error rather than silently disappearing in a positional zip.
    ids = _by_lesion_id(instance, "lesion_id", integer=True)
    ids_match_metadata = all(key == value for key, value in ids.items())
    if len(ids) != result["lesion_count"] or not ids_match_metadata:
        raise ValueError("Lesion IDs must match meta.id and the declared lesion_count.")
    fields = {
        "cacs": _by_lesion_id(instance, "lesion_score"),
        "volume_mm3": _by_lesion_id(instance, "lesion_volume"),
        "max_hu": _by_lesion_id(instance, "lesion_max_hu"),
    }
    for name, values in fields.items():
        if set(values) != set(ids):
            raise ValueError(f"Lesion {name} IDs do not match the registered lesion IDs.")
    result["lesions"] = [
        dict(id=lesion_id, **{name: values[lesion_id] for name, values in fields.items()})
        for lesion_id in sorted(ids)
    ]
    _validate_totals(result)

    return result


def _validate_totals(measurements: dict) -> None:
    """Reject inconsistent exports while preserving every supplied measurement."""
    # The scoring core assigns each qualifying 2D region to exactly one lesion.
    # Its lesion contributions must therefore reproduce the scan-level totals.
    lesions = measurements["lesions"]
    for field, summary in [("cacs", "cacs"), ("volume_mm3", "calcium_volume_mm3")]:
        if any(lesion[field] < 0 for lesion in lesions):
            raise ValueError(f"Lesion {field} must be non-negative.")
        lesion_total = sum(lesion[field] for lesion in lesions)
        if not _totals_match(lesion_total, measurements[summary]):
            raise ValueError(f"Lesion {field} sum disagrees with {summary}.")

    # Eligible calcium is a subset of the unchanged predicted segmentation.
    # An empty eligible set with a nonempty raw segmentation is valid.
    calcium_volume = measurements["calcium_volume_mm3"]
    segmentation_volume = measurements["segmentation_volume_mm3"]
    volume_exceeds_segmentation = calcium_volume > segmentation_volume
    difference_is_roundoff = _totals_match(calcium_volume, segmentation_volume)
    if volume_exceeds_segmentation and not difference_is_roundoff:
        raise ValueError("Eligible calcium volume exceeds raw segmentation volume.")



class TID1500SRGenerator(Module):
    # The SR inherits patient/study context from the source CT. The output type
    # marks it as TID 1500 for downstream selection by DataOrganizer.
    # MHub invokes this task separately for every instance, resolving its own
    # source CT, SEG evidence, measurements and output path each time.
    @IO.Instance()
    @IO.Input("source_dicom", "dicom:mod=ct", the="source CT DICOM series")
    @IO.Input("segmentation", "dicomseg:mod=seg", the="DICOM SEG evidence for the report")
    @IO.Output(
        "out_data",
        "DeepCAC2.tid1500.dcm",
        "dicom:mod=sr:tid=1500",
        data="source_dicom",
        the="TID 1500 measurement report",
    )
    def task(
        self,
        instance: Instance,
        source_dicom: InstanceData,
        segmentation: InstanceData,
        out_data: InstanceData,
    ) -> None:
        # Read the native MHub outputs; JSON is an independent export artifact.
        values = measurements_from_outputs(instance)
        report = build_tid1500_report(
            source_images=load_dicom_series(source_dicom.abspath),
            segmentation=pydicom.dcmread(segmentation.abspath),
            measurements=values,
        )
        # Explicit writing ensures the generated dataset gets a standard DICOM
        # file preamble instead of preserving an input byte layout.
        report.save_as(out_data.abspath, write_like_original=False)
