
"""Agatston scoring and calcium quantification for DeepCAC2.

All calculations use the original CT and unchanged segmentation.
Arrays are ordered (z, y, x); spacing is ordered (x, y, z).

Agatston scoring uses 2D 4-connectivity. Optional lesion analysis
groups score-qualifying components into 3D connected clusters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy import ndimage as ndi


SCORING_VERSION = "2.0.0"

HU_THRESHOLD = 130
MIN_AREA_MM2 = 1.0
REFERENCE_THICKNESS_MM = 3.0

CONNECTIVITY_2D = ndi.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class _Component:
    """One qualifying 2D component and its unnormalized score."""

    voxel: tuple[int, int, int]
    n_pixels: int
    max_hu: float
    raw_cacs: float


@dataclass
class _ScoringData:
    """Shared intermediate results for scoring and lesion analysis."""

    components: list[_Component]
    qualifying_mask: np.ndarray
    spacing: tuple[float, float, float]


def _extract_components(
    ct: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float],
) -> _ScoringData:
    """Identify and score qualifying 2D calcium components once."""
    ct = np.asarray(ct)
    mask = np.asarray(mask)

    if ct.ndim != 3 or mask.ndim != 3 or ct.shape != mask.shape:
        raise ValueError("CT and mask must have identical 3D shapes.")

    if len(spacing) != 3:
        raise ValueError("Spacing must contain three values.")

    spacing = tuple(float(s) for s in spacing)

    if not all(isfinite(s) and s > 0 for s in spacing):
        raise ValueError("Spacing must contain positive finite values.")

    pixel_area = spacing[0] * spacing[1]

    # Apply the HU threshold without changing the original segmentation.
    candidate = (mask > 0) & (ct >= HU_THRESHOLD)
    qualifying = np.zeros_like(candidate, dtype=bool)

    components = []

    # Components are independently identified on each axial slice.
    for z in range(candidate.shape[0]):
        labels, _ = ndi.label(candidate[z], CONNECTIVITY_2D)

        for label_id, bounds in enumerate(
            ndi.find_objects(labels), start=1
        ):
            if bounds is None:
                continue

            region = labels[bounds] == label_id
            n_pixels = int(np.count_nonzero(region))
            area = n_pixels * pixel_area

            if area < MIN_AREA_MM2:
                continue

            # Retain only voxels belonging to qualifying components.
            qualifying[z][bounds] |= region

            max_hu = float(ct[z][bounds][region].max())
            density = 1 + int(
                np.searchsorted([200, 300, 400], max_hu, side="right")
            )

            # Record a representative voxel for subsequent 3D grouping.
            y, x = np.argwhere(region)[0]
            voxel = (
                z,
                bounds[0].start + int(y),
                bounds[1].start + int(x),
            )

            components.append(
                _Component(
                    voxel=voxel,
                    n_pixels=n_pixels,
                    max_hu=max_hu,
                    raw_cacs=area * density,
                )
            )

    return _ScoringData(components, qualifying, spacing)


def _summarize_agatston(data: _ScoringData) -> dict:
    """Calculate the standard score from previously identified regions."""
    factor = data.spacing[2] / REFERENCE_THICKNESS_MM

    return {
        "cacs": float(sum(
            c.raw_cacs * factor for c in data.components
        )),
        "n_regions": len(data.components),
        "eligible_voxels": sum(
            c.n_pixels for c in data.components
        ),
    }


def calculate_agatston(
    ct: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float],
) -> dict[str, float | int]:
    """Calculate spacing-normalized Agatston score.

    Returns the score and qualifying-region counts:
        cacs, n_regions, eligible_voxels.
    """
    data = _extract_components(ct, mask, spacing)
    return _summarize_agatston(data)


def _identify_lesions(
    data: _ScoringData,
    connectivity: int,
) -> list[dict]:
    """Group qualifying 2D components into 3D calcium clusters.

    This grouping does not influence the Agatston calculation.
    Each 2D component belongs entirely to one 3D connected cluster.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("Connectivity must be 6, 18 or 26.")

    if not data.components:
        return []

    structure = ndi.generate_binary_structure(
        3, {6: 1, 18: 2, 26: 3}[connectivity]
    )
    labels, n_lesions = ndi.label(data.qualifying_mask, structure)

    spacing_factor = data.spacing[2] / REFERENCE_THICKNESS_MM
    voxel_volume = float(np.prod(data.spacing))

    lesions = [
        {
            "id": i,
            "cacs": 0.0,
            "n_regions_2d": 0,
            "eligible_voxels": 0,
            "volume_mm3": 0.0,
            "max_hu": None,
        }
        for i in range(1, n_lesions + 1)
    ]

    # Aggregate the already calculated 2D contributions by cluster.
    for component in data.components:
        lesion_id = int(labels[component.voxel])
        lesion = lesions[lesion_id - 1]

        lesion["cacs"] += component.raw_cacs * spacing_factor
        lesion["n_regions_2d"] += 1
        lesion["eligible_voxels"] += component.n_pixels
        lesion["max_hu"] = max(
            component.max_hu,
            lesion["max_hu"]
            if lesion["max_hu"] is not None
            else component.max_hu,
        )

    for lesion in lesions:
        lesion["volume_mm3"] = (
            lesion["eligible_voxels"] * voxel_volume
        )

    return lesions


def analyze_calcium(
    ct: np.ndarray,
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    lesion_connectivity: int = 26,
) -> dict:
    """Calculate all calcium scores, volumes and 3D lesion measurements.

    Scoring components are extracted only once. Scores use slice spacing
    divided by 3 mm; volumes use physical voxel spacing.
    """
    if lesion_connectivity not in (6, 18, 26):
        raise ValueError("Connectivity must be 6, 18 or 26.")

    data = _extract_components(ct, mask, spacing)
    summary = _summarize_agatston(data)

    voxel_volume = float(np.prod(data.spacing))

    lesions = _identify_lesions(data, lesion_connectivity)

    return {
        **summary,
        "calcium_volume_mm3": (
            summary["eligible_voxels"] * voxel_volume
        ),
        "segmentation_volume_mm3": (
            int(np.count_nonzero(np.asarray(mask) > 0)) * voxel_volume
        ),
        "lesion_count": len(lesions),
        "lesion_connectivity": lesion_connectivity,
        "lesions": lesions,
    }


def classify_risk(cacs: float) -> int:
    """Return risk category 0–3 using the unrounded Agatston score."""
    if not isfinite(cacs) or cacs < 0:
        raise ValueError("CACS must be finite and non-negative.")

    if cacs == 0:
        return 0
    if cacs <= 100:
        return 1
    if cacs <= 300:
        return 2
    return 3
