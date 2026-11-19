# DeepCAC2

All bundled workflows use `CalciumAnalysisCalculator` and scoring version **2.0.0**.
Inference and the predicted segmentation are unchanged. The calculation uses
CT intensities in HU on the original image grid and rejects mismatched CT/mask
size, spacing, origin, or direction.

On each axial slice, candidate voxels satisfy `mask > 0` and `CT >= 130 HU`.
Components use 4-connectivity and must occupy at least 1 mm2. Their area is
weighted by peak attenuation (factors 1/2/3/4 at 130/200/300/400 HU), then
multiplied by physical slice spacing / 3 mm. Scores retain full precision;
risk groups use the unrounded score: 0, (0,100], (100,300], and >300.

Eligible components are grouped into 3D lesions using 26-connectivity by
default. Configure `CalciumAnalysisCalculator.lesion_connectivity` as 6, 18,
or 26; this changes lesion grouping but never the score. Lesion IDs follow
spatial labeling order, rather than the previous largest-volume-first order.

JSON is produced by MHub's `ReportExporter`: `DeepCAC.report.json` contains
full-precision scores, risk category, eligible calcium volume, raw segmentation
volume, lesion count, individual lesion measurements, and `Scoring Version`.
The DICOM workflows export per-series and combined reports; `pid_nifti` exports
a combined report. There is no separate `DeepCAC2.measurements.json` or dataset
`info.json` export.

The calculator uses only its declared NIfTI CT and CAC mask inputs. Scores
and volumes use physical voxel spacing; no DICOM headers are read by this
module. Only the primary spacing-normalized score is calculated. Dataset-only
schema fields, duplicate score aliases, nominal thickness placeholders, and
alternative unnormalized/thickness-normalized score outputs have been removed.

Individual lesion measurements are also registered in `instance.outputData`
and included as parallel lists in the compact per-case and combined reports:

| MHub output selector | Report label |
| --- | --- |
| `lesion_id` | Lesion IDs |
| `lesion_volume` | Individual Lesion Volumes (mm3) |
| `lesion_score` | Individual Lesion Agatston Scores |
| `lesion_max_hu` | Individual Lesion Maximum HU |
| `lesion_eligible_voxels` | Individual Lesion Eligible Voxel Counts |
| `lesion_region_count` | Individual Lesion 2D Region Counts |

All lists follow spatial lesion-ID order; corresponding entries describe
the same lesion. Each registered output carries the shared
lesion ID in `meta.id`, its unit, and the CAC segmentation metadata. For example,
`lesion_volume:id=2` selects the second lesion's volume. Report configurations
select repeated outputs with `aggregate: list`. Cases with no eligible lesions
produce empty lists. Lesion scores use the primary spacing normalization.

Calcium volume includes only score-eligible voxels; segmentation volume
includes every positive voxel of the unchanged prediction. Therefore a
nonempty segmentation can have zero calcium score, volume, and lesions.
Empty masks produce zero measurements and an empty lesion list.

The DICOM workflow also exports the HEART/CAC DICOM SEG and
`DeepCAC2.tid1500.dcm`. The SR contains global calcium measurements, a separate
raw-segmentation volume group referencing the CAC segment, and individually
tracked lesion measurement groups. Lesions do not claim separate spatial
segments. A CAC-negative SEG containing only HEART uses image-level zero
measurements. Nonzero measurements or raw segmentation volume without a CAC
segment are rejected. Reports are complete, preliminary, and unverified.

The SR adapter declares the source CT and DICOM SEG as explicit inputs. A
negative case may provide a HEART-only SEG; the module never searches for an
undeclared SEG or substitutes HEART for CAC. It reads MHub outputs directly,
joining individual lesion measurements by `meta.id`. It requires exactly one of each summary
output and one score, volume, and maximum-HU output for each registered lesion
ID. Scoring version and actual configured lesion connectivity are published by
the calculator and included in the SR algorithm identification. The fixed segmentation weights belong to the model/container release and
are not published as a separate measurement. Risk
category parameters state the boundaries for the unrounded score. Negative
tracking identifiers are scoped to the source CT series, so multiple series
within the same study remain distinct. CAC segment selection uses both the
SNOMED CT scheme and code and rejects ambiguous duplicate CAC segments.
The adapter validates the lesion count, risk category, and score/volume totals
before building the report. SR generation uses MHub objects without reading JSON
reports or intermediate measurement files.

Batch inputs are processed separately by MHub's `@IO.Instance()` decorator.
Each instance has its own output collection, so lesion IDs may repeat across
cases. The SR adapter expects one calcium-analysis result set per instance;
multiple competing result sets within the same instance are rejected as
ambiguous. Its consistency checks do not change the scoring algorithm or
round measurements. Score/volume totals allow floating-point roundoff
(`rel_tol=1e-9`, `abs_tol=1e-9`); missing or inconsistent outputs prevent SR
generation rather than being silently corrected.

This intentionally changes historical scores and replaces raw-mask plaque
volume/count fields with explicitly named calcium/segmentation measurements.
The obsolete Agatston and raw-mask plaque calculators and their helpers have
been removed. The active calculator publishes eligible lesion measurements
and raw segmentation volume. MHub output classes are defined alongside the calculator in
`utils/CalciumAnalysisCalculator.py`; the scoring algorithm lives in
`src/scoring.py`.
Existing reference results must be regenerated before comparing against
scoring version 2.0.0.

The scoring core and baseline tests were ported from
`DeepCAC2_pub/deepcac2_dataset/src/deepcac2_dataset/scoring.py`
(repository HEAD `e5ff29b496d582f6e149dda2d2ef621ee6676ab2` at integration).
The local port adds deferred annotations for the older MHub Python runtime
and an explicit scoring version; it does not require the dataset package.
Unused dataset reporting fields and alternative normalizations have been
removed without changing the primary scoring and lesion calculations.

## Build and export the local working tree

Run from the repository root. Both the legacy Docker builder and BuildKit are supported.
This includes uncommitted source changes; the default build still imports from
`MHUB_MODELS_REPO` (or upstream main). Only `models/deepcac2` is sent as the build context. Its `.dockerignore`
excludes test data, local weights, and Python caches; image archives elsewhere
in the repository are outside the context entirely.
Weights are downloaded by the Dockerfile as usual.

```bash
set -euo pipefail

docker build \
  --file models/deepcac2/dockerfiles/Dockerfile \
  --build-arg MHUB_LOCAL_MODEL=true \
  --tag mhubai/deepcac2:scoring-v2 \
  models/deepcac2

docker image save mhubai/deepcac2:scoring-v2 | gzip > mhubai_deepcac2_scoring-v2_20261028.tar.gz
```

The new archive is written beside the existing archive in the repository root.

## DICOM-to-DICOM workflow

`dicom_dicom` explicitly selects DICOM input, dcm2niix conversion, inference,
shared calcium measurements, DICOM SEG, TID 1500 SR, JSON reports, and the CAC
NRRD. It has the same output set as `default`.

```bash
mkdir -p output_data

docker run --rm --gpus all \
  --volume /absolute/path/to/input-dicom:/app/data/input_data:ro \
  --volume "$PWD/output_data:/app/data/output_data" \
  mhubai/deepcac2:scoring-v2 --workflow dicom_dicom
```

## DICOM validation

Build the independent validation image from the repository root:

```bash
docker build \
  --file models/deepcac2/dockerfiles/validate.Dockerfile \
  --tag deepcac2-dicom-validate \
  models/deepcac2/dockerfiles
```

Mount one organized case directory and validate both derived objects:

```bash
REFERENCE_DIR="$PWD/models/deepcac2/test/default/reference/<case-id>"

docker run --rm --volume "$REFERENCE_DIR:/data:ro" \
  deepcac2-dicom-validate \
  dciodvfy -new /data/DeepCAC2.seg.dcm

docker run --rm --volume "$REFERENCE_DIR:/data:ro" \
  deepcac2-dicom-validate \
  dciodvfy -new /data/DeepCAC2.tid1500.dcm

docker run --rm --volume "$REFERENCE_DIR:/data:ro" \
  deepcac2-dicom-validate \
  validate-sr -checktemplateid -checkcontentitemorder \
  -checkambiguoustemplate /data/DeepCAC2.tid1500.dcm

docker run --rm --volume "$REFERENCE_DIR:/data:ro" \
  deepcac2-dicom-validate \
  dsrdump +Pt +Pc /data/DeepCAC2.tid1500.dcm
```

`dciodvfy` checks the IOD and data-element constraints, PixelMed checks the SR
template and context groups, and `dsrdump` provides an independently parsed
view of the content tree. Warnings require review; a zero exit code alone does
not establish semantic correctness or pixel equivalence with the source mask.

### Validation status for scoring 2.0.0

The existing positive sample was checked against the dataset scorer: CACS
101.76086849106444, seven lesions. The converted NIfTI gives
101.7608642578125 (floating-point spacing precision) after aligning the
historical mask by an exact axis flip. The MHub scoring, JSON and SR modules
were also executed together with that aligned sample and highdicom 0.22.0.
The historical mask and a newly converted CT must not be paired as unaligned
arrays; the new geometry checks reject that combination.

`dciodvfy` reported no IOD errors on the generated SR, with warnings for absent
source StudyID/StudyTime and private measurement codes. PixelMed's installed
`DicomSRDescriptionsCompiled.xsl` selects the spatial TID 1411 template when
*any* measurement group contains a segment reference, then applies it to all
measurement groups. It also invokes TID 1501 across the groups. Consequently,
this mixed TID 1411/TID 1501 report produces false missing-ROI diagnostics and
ambiguous matches on non-spatial groups. The report explicitly encodes TID 1411
only for raw segmentation volume and TID 1501 for global and lesion groups.
The validator was not modified to suppress these diagnostics; independent SR
template validation is **not clean**. A zero validator exit status must not be
interpreted as a pass. The installed validator needs template selection scoped
to individual groups before it can establish conformance for this report.
