"""Calculate calcium measurements once for all MHub exports."""
import numpy as np
import SimpleITK as sitk
from mhubio.core import IO, Instance, InstanceData, Meta, Module, OutputDataCollection, ValueOutput, ClassOutput

from ..src.scoring import (
    analyze_calcium, classify_risk, SCORING_VERSION,
)


@ValueOutput.Name("scoring_version")
@ValueOutput.Label("ScoringVersion")
@ValueOutput.Type(str)
@ValueOutput.Description("Version of the calcium scoring algorithm.")
class ScoringVersion(ValueOutput):
    pass


@ValueOutput.Name("lesion_connectivity")
@ValueOutput.Label("LesionConnectivity")
@ValueOutput.Type(int)
@ValueOutput.Description("3D connectivity used to group score-eligible calcium into lesions.")
class LesionConnectivity(ValueOutput):
    pass


@ValueOutput.Name("cacs")
@ValueOutput.Label("CACS")
@ValueOutput.Type(float)
@ValueOutput.Description("CAC score, normalized by slice spacing / 3 mm.")
class Cacs(ValueOutput):
    pass


@ValueOutput.Name("calcium_volume")
@ValueOutput.Label("calcium_volume")
@ValueOutput.Type(float)
@ValueOutput.Meta(Meta(unit="mm3"))
@ValueOutput.Description("Volume of score-eligible calcium in mm3.")
class CalciumVolume(ValueOutput):
    pass


@ValueOutput.Name("segmentation_volume")
@ValueOutput.Label("segmentation_volume")
@ValueOutput.Type(float)
@ValueOutput.Meta(Meta(unit="mm3"))
@ValueOutput.Description("Volume of the unchanged predicted segmentation in mm3.")
class SegmentationVolume(ValueOutput):
    pass


@ValueOutput.Name("lesion_count")
@ValueOutput.Label("lesion_count")
@ValueOutput.Type(int)
@ValueOutput.Meta(Meta(unit="{lesions}"))
@ValueOutput.Description("Number of 3D connected score-eligible lesions.")
class LesionCount(ValueOutput):
    pass


@ValueOutput.Name("lesion_id")
@ValueOutput.Label("LesionId")
@ValueOutput.Type(int)
@ValueOutput.Meta(Meta(unit="1"))
@ValueOutput.Description("Identifier of one score-eligible 3D lesion.")
class LesionId(ValueOutput):
    pass


@ValueOutput.Name("lesion_volume")
@ValueOutput.Label("LesionVolume")
@ValueOutput.Type(float)
@ValueOutput.Meta(Meta(unit="mm3"))
@ValueOutput.Description("Physical volume of one score-eligible 3D lesion in mm3.")
class LesionVolume(ValueOutput):
    pass


@ValueOutput.Name("lesion_score")
@ValueOutput.Label("LesionScore")
@ValueOutput.Type(float)
@ValueOutput.Meta(Meta(unit="1"))
@ValueOutput.Description("Spacing-normalized CAC score contribution of one 3D lesion.")
class LesionScore(ValueOutput):
    pass


@ValueOutput.Name("lesion_max_hu")
@ValueOutput.Label("LesionMaxHU")
@ValueOutput.Type(float)
@ValueOutput.Meta(Meta(unit="[hnsf'U]"))
@ValueOutput.Description("Maximum attenuation of one score-eligible 3D lesion in HU.")
class LesionMaxHU(ValueOutput):
    pass


@ValueOutput.Name("lesion_eligible_voxels")
@ValueOutput.Label("LesionEligibleVoxels")
@ValueOutput.Type(int)
@ValueOutput.Meta(Meta(unit="1"))
@ValueOutput.Description("Number of score-eligible voxels in one 3D lesion.")
class LesionEligibleVoxels(ValueOutput):
    pass


@ValueOutput.Name("lesion_region_count")
@ValueOutput.Label("LesionRegionCount")
@ValueOutput.Type(int)
@ValueOutput.Meta(Meta(unit="1"))
@ValueOutput.Description("Number of qualifying 2D regions in one 3D lesion.")
class LesionRegionCount(ValueOutput):
    pass


@ClassOutput.Name('risk_category')
@ClassOutput.Label('RiskCategory')
@ClassOutput.Description('CAC risk category.')
@ClassOutput.Class(0, 'Very Low', the='No score-eligible CAC: CACS = 0.')
@ClassOutput.Class(1, 'Low', the='0 < CACS <= 100.')
@ClassOutput.Class(2, 'Moderate', the='100 < CACS <= 300.')
@ClassOutput.Class(3, 'High', the='CACS > 300.')
class RiskCategory(ClassOutput):
    pass


def _validate_geometry(ct: sitk.Image, mask: sitk.Image) -> None:
    """Require CT and mask voxels to share the same physical grid."""
    if ct.GetDimension() != 3 or mask.GetDimension() != 3 or ct.GetSize() != mask.GetSize():
        raise ValueError("CT and CAC mask must have matching 3D dimensions.")
    for name, tolerance in [("Spacing", 1e-3), ("Origin", 1e-3), ("Direction", 1e-5)]:
        ct_geometry = getattr(ct, "Get" + name)()
        mask_geometry = getattr(mask, "Get" + name)()
        if not np.allclose(ct_geometry, mask_geometry, rtol=0, atol=tolerance):
            raise ValueError("CT and CAC mask geometry mismatch: " + name)


@IO.Config("lesion_connectivity", int, 26, the="3D lesion connectivity: 6, 18 or 26")
class CalciumAnalysisCalculator(Module):
    lesion_connectivity: int

    @IO.Instance()
    @IO.Input("image", "nifti:mod=ct")
    @IO.Input("cac", "nrrd:mod=seg:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION")
    @IO.OutputData("connectivity", LesionConnectivity, data="cac")
    @IO.OutputData("scoring_version", ScoringVersion, data="cac")
    @IO.OutputData("cacs", Cacs, data="cac")
    @IO.OutputData("risk_category", RiskCategory, data="cac")
    @IO.OutputData("calcium_volume", CalciumVolume, data="cac")
    @IO.OutputData("segmentation_volume", SegmentationVolume, data="cac")
    @IO.OutputData("lesion_count", LesionCount, data="cac")
    @IO.OutputDatas("lesion_id_outputs", LesionId)
    @IO.OutputDatas("lesion_volume_outputs", LesionVolume)
    @IO.OutputDatas("lesion_score_outputs", LesionScore)
    @IO.OutputDatas("lesion_max_hu_outputs", LesionMaxHU)
    @IO.OutputDatas("lesion_eligible_voxels_outputs", LesionEligibleVoxels)
    @IO.OutputDatas("lesion_region_count_outputs", LesionRegionCount)
    def task(
        self, instance: Instance, image: InstanceData, cac: InstanceData,
        cacs: Cacs, risk_category: RiskCategory,
        calcium_volume: CalciumVolume, segmentation_volume: SegmentationVolume,
        lesion_count: LesionCount, scoring_version: ScoringVersion,
        connectivity: LesionConnectivity,
        lesion_id_outputs: OutputDataCollection,
        lesion_volume_outputs: OutputDataCollection,
        lesion_score_outputs: OutputDataCollection,
        lesion_max_hu_outputs: OutputDataCollection,
        lesion_eligible_voxels_outputs: OutputDataCollection,
        lesion_region_count_outputs: OutputDataCollection,
    ) -> None:
        ct = sitk.ReadImage(image.abspath)
        mask = sitk.ReadImage(cac.abspath)
        _validate_geometry(ct, mask)

        # One calculation supplies the scalar and individual lesion outputs.
        result = analyze_calcium(
            sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(mask), ct.GetSpacing(),
            lesion_connectivity=self.lesion_connectivity,
        )
        # Scalar outputs are attached to this instance by the IO decorators.
        cacs.value = result["cacs"]
        risk_category.value = classify_risk(result["cacs"])
        scoring_version.value = SCORING_VERSION
        connectivity.value = result["lesion_connectivity"]
        calcium_volume.value = result["calcium_volume_mm3"]
        segmentation_volume.value = result["segmentation_volume_mm3"]
        lesion_count.value = result["lesion_count"]
        # Publish each lesion's measurements as repeated MHub outputs. The
        # mapping below pairs an output schema/collection with its scoring key.
        # A shared meta.id links the fields for one lesion across reports and SR;
        # segmentation metadata supplies provenance, and each schema keeps its unit.
        for lesion in result["lesions"]:
            for schema, collection, key in [
                (LesionId, lesion_id_outputs, "id"),
                (LesionVolume, lesion_volume_outputs, "volume_mm3"),
                (LesionScore, lesion_score_outputs, "cacs"),
                (LesionMaxHU, lesion_max_hu_outputs, "max_hu"),
                (LesionEligibleVoxels, lesion_eligible_voxels_outputs, "eligible_voxels"),
                (LesionRegionCount, lesion_region_count_outputs, "n_regions_2d"),
            ]:
                output = schema()
                output.value = lesion[key]
                output.meta = cac.type.meta + output.meta + Meta(id=str(lesion["id"]))
                collection.add(output)
