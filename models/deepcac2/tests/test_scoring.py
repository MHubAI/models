"""Unit tests for the DeepCAC2 Agatston scoring implementation."""

import numpy as np
import pytest

from models.deepcac2.src.scoring import (
    calculate_agatston,
    classify_risk,
    analyze_calcium
)


def test_validation_notebook_example():
    """Reproduce the independently validated synthetic example."""
    ct = np.zeros((4, 10, 10), dtype=np.float32)
    mask = np.zeros_like(ct, dtype=bool)

    # Three qualifying components with density factors 2, 3 and 4.
    for z, hu in enumerate([200, 350, 410]):
        mask[z, 1:3, 1:3] = True
        ct[z, 1:3, 1:3] = hu

    # Excluded: area below 1 mm².
    mask[0, 6, 6:8] = True
    ct[0, 6, 6:8] = 450

    # Excluded: attenuation below 130 HU.
    mask[0, 6:8, 1:3] = True
    ct[0, 6:8, 1:3] = 100

    result = calculate_agatston(ct, mask, (0.5, 0.5, 3.0))

    assert result["cacs"] == pytest.approx(9.0)
    assert result["n_regions"] == 3
    assert result["eligible_voxels"] == 12


@pytest.mark.parametrize(
    "hu, expected",
    [
        (129, 0),
        (130, 1),
        (199, 1),
        (200, 2),
        (299, 2),
        (300, 3),
        (399, 3),
        (400, 4),
    ],
)
def test_hu_density_thresholds(hu, expected):
    """Verify attenuation thresholds and density factors."""
    ct = np.full((1, 2, 2), hu)
    mask = np.ones_like(ct)

    result = calculate_agatston(ct, mask, (0.5, 0.5, 3.0))

    assert result["cacs"] == pytest.approx(expected)


def test_four_connectivity():
    """Diagonally touching pixels must form separate components."""
    ct = np.zeros((1, 3, 3))
    mask = np.zeros_like(ct)

    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 1

    ct[0, 0, 0] = 200
    ct[0, 1, 1] = 400

    result = calculate_agatston(ct, mask, (1.0, 1.0, 3.0))

    assert result["n_regions"] == 2
    assert result["cacs"] == pytest.approx(6.0)


def test_minimum_area():
    """The minimum area must be inclusive at exactly 1 mm²."""
    ct = np.full((1, 3, 3), 200)
    mask = np.zeros_like(ct)

    mask[0, 0, 0:3] = 1  # 0.75 mm².
    assert calculate_agatston(
        ct, mask, (0.5, 0.5, 3.0)
    )["cacs"] == 0

    mask[0, 1, 0] = 1  # Now 1.0 mm².
    assert calculate_agatston(
        ct, mask, (0.5, 0.5, 3.0)
    )["cacs"] == pytest.approx(2.0)


def test_slice_spacing_normalization():
    """Halving the z-spacing must halve the score."""
    ct = np.full((1, 2, 2), 200)
    mask = np.ones_like(ct)

    standard = calculate_agatston(ct, mask, (0.5, 0.5, 3.0))
    thin = calculate_agatston(ct, mask, (0.5, 0.5, 1.5))

    assert standard["cacs"] == pytest.approx(2.0)
    assert thin["cacs"] == pytest.approx(1.0)


def test_empty_mask():
    """Empty masks must produce zero scores and counts."""
    ct = np.full((2, 4, 4), 400)
    mask = np.zeros_like(ct)

    assert calculate_agatston(ct, mask, (1, 1, 3)) == {
        "cacs": 0.0,
        "n_regions": 0,
        "eligible_voxels": 0,
    }


def test_inputs_unchanged():
    """Scoring must never modify CT or segmentation arrays."""
    ct = np.full((1, 2, 2), 200)
    mask = np.ones_like(ct)

    original_ct = ct.copy()
    original_mask = mask.copy()

    calculate_agatston(ct, mask, (1, 1, 3))

    np.testing.assert_array_equal(ct, original_ct)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize(
    "cacs, category",
    [
        (0, 0),
        (0.001, 1),
        (1, 1),
        (100, 1),
        (100.001, 2),
        (300, 2),
        (300.001, 3),
    ],
)
def test_risk_categories(cacs, category):
    """Verify classification using unrounded CACS."""
    assert classify_risk(cacs) == category

def test_analyze_empty_mask():
    """An empty segmentation produces no calcium or lesions."""
    ct = np.full((2, 4, 4), 400)
    mask = np.zeros_like(ct)

    result = analyze_calcium(
        ct, mask, (1.0, 1.0, 3.0),
    )

    assert result["cacs"] == 0
    assert result["calcium_volume_mm3"] == 0
    assert result["segmentation_volume_mm3"] == 0
    assert result["lesion_count"] == 0
    assert result["lesions"] == []


def test_analyze_3d_lesion_connectivity():
    """3D grouping must not alter 2D Agatston scores."""
    ct = np.zeros((2, 4, 4), dtype=np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)

    # Two diagonally adjacent voxels in consecutive slices.
    ct[0, 0, 0] = 200
    ct[1, 1, 1] = 400
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1

    spacing = (1.0, 1.0, 3.0)

    connected = analyze_calcium(
        ct, mask, spacing, lesion_connectivity=26
    )
    disconnected = analyze_calcium(
        ct, mask, spacing, lesion_connectivity=6
    )

    # Scoring is independent of 3D lesion connectivity.
    assert connected["cacs"] == pytest.approx(6.0)
    assert disconnected["cacs"] == pytest.approx(6.0)

    assert connected["n_regions"] == 2
    assert connected["lesion_count"] == 1
    assert disconnected["lesion_count"] == 2

    assert connected["calcium_volume_mm3"] == pytest.approx(6.0)
    assert connected["lesions"][0]["cacs"] == pytest.approx(6.0)
    assert connected["lesions"][0]["volume_mm3"] == pytest.approx(6.0)


def test_spacing_normalization_and_lesion_totals():
    ct = np.full((2, 4, 4), 400.0)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[:, :2, :2] = 1
    mask[0, 3, 3] = 1  # Raw mask voxel excluded by the 1 mm2 area rule.
    result = analyze_calcium(ct, mask, (0.5, 0.5, 1.5))
    assert result["cacs"] == 4
    assert result["segmentation_volume_mm3"] > result["calcium_volume_mm3"]
    assert sum(l["cacs"] for l in result["lesions"]) == pytest.approx(result["cacs"])
    assert sum(l["volume_mm3"] for l in result["lesions"]) == pytest.approx(result["calcium_volume_mm3"])


@pytest.mark.parametrize("spacing", [(0, 1, 3), (1, float("nan"), 3), (1, 1, -1)])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError, match="positive finite"):
        analyze_calcium(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), spacing)
