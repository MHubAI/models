# DeepCAC2

## Workflows

Select a bundled configuration with `--workflow <name>`.

| Workflow | Input | CT preprocessing | Exported files |
| --- | --- | --- | --- |
| `default` | DICOM CT | dcm2niix, then LPS normalization | DICOM SEG, TID 1500 SR, per-series and combined JSON |
| `dicom_dicom_plastimatch` | DICOM CT | Default NiftiConverter engine (plastimatch), without LPSOrienter | Same as `default` |
| `dicom_extended` | DICOM CT | dcm2niix, then LPS normalization | Same as `default`, plus the CAC NRRD |
| `nifti_pid` | NIfTI CT named `<PatientID>.nii.gz` with numeric PatientID | LPS normalization | Combined JSON |

Each input is processed separately. Both segmentation stages and calcium
analysis use the same working CT. Workflow settings are in [config/](config/).

## Modules

The four model-specific modules below operate on MHub-managed inputs and
outputs. Configuration keys go under `modules.<ModuleName>` in the workflow.

### LPSOrienter

Input: `nifti:mod=ct`. Output: `image_lps.nii.gz`, registered as
`nifti:mod=ct:orientation=LPS`.

Creates the LPS axis ordering used for inference by permuting/flipping axes
without interpolation, preserving HU values and physical voxel locations.
Oblique directions are retained. The previous CT entry is retired so downstream
modules select the oriented image.

| Configuration | Default | Effect |
| --- | --- | --- |
| `delete_input_file` | `false` | Also delete the replaced CT file after writing the oriented image |

`default` and `dicom_extended` enable deletion of the generated intermediate
NIfTI. `nifti_pid` preserves the imported file. Source DICOM files are retained.

### DeepCACRunner

Inputs: CT (`nifti:mod=ct`) and heart segmentation (`nifti:mod=seg`) from
`NNUnetRunner`. Output: `pcac.nrrd`, registered as
`nrrd:mod=seg:model=DeepCAC2:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION`.

Runs CAC inference with the container's fixed weights. There are no
module-specific configuration options.

### CalciumAnalysisCalculator

Inputs: CT (`nifti:mod=ct`) and CAC mask
(`nrrd:mod=seg:roi=CORONARY_ARTERY+PATHOLOGIC_CALCIFICATION`). The images must
share their size and physical geometry. This module reads no DICOM headers.

| Configuration | Default | Effect |
| --- | --- | --- |
| `lesion_connectivity` | `26` | Group eligible calcium into 3D lesions using 6, 18, or 26 neighbors; does not change the total score |

Scoring version **2.0.0** uses voxels with `mask > 0` and `CT >= 130 HU`.
On each axial slice, 4-connected regions must occupy at least 1 mm². Region
area is weighted by peak attenuation (factors 1/2/3/4 at 130/200/300/400 HU)
and multiplied by physical slice spacing / 3 mm. Scores retain full precision.

Outputs are native MHub measurements consumed by the report and SR modules:

| Output selector | Meaning |
| --- | --- |
| `scoring_version` | Scoring algorithm version |
| `lesion_connectivity` | Configured 3D connectivity |
| `cacs`, `risk_category` | Score and category |
| `calcium_volume`, `segmentation_volume` | Eligible calcium and raw positive-mask volume, in mm³ |
| `lesion_count` | Number of eligible 3D lesions |
| `lesion_id`, `lesion_volume`, `lesion_score` | Repeated lesion IDs, eligible volumes, and score contributions |
| `lesion_max_hu`, `lesion_eligible_voxels`, `lesion_region_count` | Repeated lesion maximum HU, eligible voxel counts, and qualifying 2D-region counts |

Repeated outputs share `meta.id` for the same lesion; for example,
`lesion_volume:id=2` selects lesion 2's volume. Empty eligible masks produce
zero measurements and no lesion outputs. A nonempty raw mask can still have
zero eligible calcium volume, score, and lesions.

### TID1500SRGenerator

Inputs: source CT series (`dicom:mod=ct`), DICOM SEG (`dicomseg:mod=seg`),
and the calculator's native MHub measurements for the same instance.
Output: `DeepCAC2.tid1500.dcm` (`dicom:mod=sr:tid=1500`). There are no
module-specific configuration options.

Creates a TID 1500 measurement report containing global score, category,
eligible volume and lesion count; raw segmentation volume referencing the
CAC segment; and individual lesion score, volume and maximum HU. Lesions have
tracking identifiers but no separate spatial segments. A HEART-only negative
SEG produces global zero measurements without CAC or lesion groups.
Reports are complete, preliminary, and unverified.

### Standard MHub modules

The workflows also configure these framework modules. The table lists the
settings used here rather than the framework's complete configuration API.

| Module | Inputs → outputs | Settings used here |
| --- | --- | --- |
| `DicomImporter` | DICOM files → CT series instances | `meta` maps Modality and PatientID into file metadata |
| `FileStructureImporter` | Named NIfTI files → CT instances | `input_dir`, `import_id`, `structures` define file discovery and PatientID extraction |
| `NiftiConverter` | DICOM CT → NIfTI CT | `engine: dcm2niix` in `default` and `dicom_extended`; omitted for plastimatch |
| `NNUnetRunner` | NIfTI CT → NIfTI HEART mask | `nnunet_task: Task400_OPEN_HEART_1FOLD`, `nnunet_model: 3d_lowres`, `folds: all`, `roi: HEART` |
| `DsegConverter` | HEART/CAC masks and source CT → DICOM SEG | `source_segs`, `target_dicom`, `model_name`, `body_part_examined`, `converted_file_name`, `skip_empty_slices` |
| `ReportExporter` | Native measurements and metadata → JSON | `includes` selects fields; `aggregate: list` collects lesions; `globalreport: true` produces the combined report; `meta` distinguishes report scope |
| `DataOrganizer` | Generated files → exported directory | `targets` defines file selection and destination paths |

## Exported outputs

DICOM workflows export:

```text
DeepCAC.report.json
<SeriesInstanceUID>/
  DeepCAC2.seg.dcm
  DeepCAC2.tid1500.dcm
  DeepCAC.report.json
```

The root JSON is a list of per-series reports. `dicom_extended` additionally
exports `<SeriesInstanceUID>/nrrd/cac.seg.nrrd`, containing the unchanged CAC
prediction. `nifti_pid` exports only the combined JSON, identifies cases by
PatientID instead of SeriesInstanceUID, and includes score/category descriptions.
DICOM PatientID comes from the imported CT metadata and is empty if unavailable.

The SEG contains HEART and unchanged CAC masks with source CT references.
Empty slices are skipped, and an empty CAC mask may be omitted. SEG segment
numbers are distinct from lesion IDs in JSON and SR.

The DICOM-workflow JSON fields are:

| JSON field | Type and meaning |
| --- | --- |
| `SeriesInstanceUID` | String; original CT series UID, not the generated SEG/SR series UID |
| `PatientID` | String from source CT metadata; empty if unavailable |
| `Scoring Version` | String; calcium-analysis algorithm version |
| `Lesion Connectivity` | Integer; configured 3D connectivity (6, 18, or 26) |
| `Predicted CAC Score` | Float; unrounded score normalized by slice spacing / 3 mm |
| `Predicted Risk Group` | Integer 0–3; 0, (0,100], (100,300], >300 using the unrounded score |
| `Calcium Volume (mm3)` | Float; volume of score-eligible calcium |
| `Segmentation Volume (mm3)` | Float; volume of every positive voxel in the unchanged CAC prediction, excluding HEART |
| `Number of Lesions` | Integer; 3D connected components of score-eligible calcium |
| `Lesion IDs` | List of local integer IDs |
| `Individual Lesion Volumes (mm3)` | List of eligible volumes |
| `Individual Lesion CAC Scores` | List of contributions to the global CACS |
| `Individual Lesion Maximum HU` | List of maximum HU values over eligible voxels |
| `Individual Lesion Eligible Voxel Counts` | List of integer eligible-voxel counts |
| `Individual Lesion 2D Region Counts` | List of integer qualifying 2D-component counts |

Corresponding positions in all six lesion lists refer to the same lesion. IDs
are local to a result and are not stable longitudinal identifiers or SEG segment
numbers; image-axis ordering can change their numbering. Empty eligible masks
produce zero measurements and six empty lists. Numeric values are not rounded
for export. Total eligible voxels, total qualifying 2D regions, and largest
lesion volume can be derived from the lists; they are not duplicated as fields.

## Python tests

From the repository root, run the CPU test suite in an isolated uv environment:

```bash
uv run --no-project --python 3.10 \
  --with pytest --with 'numpy==1.26.4' --with scipy \
  --with 'mhubio @ git+https://github.com/MHubAI/mhubio.git' \
  --with 'segdb @ git+https://github.com/MHubAI/SegDB.git' \
  python -m pytest -q models/deepcac2/tests
```

The first run downloads MHubIO and its dependencies; subsequent runs reuse uv's
cache. Tests cover scoring, orientation, MHub integration, and DICOM SR generation.
They do not run model inference or require a GPU or model weights.

## Citation

If you use DeepCAC2, please cite:

Nürnberg L, Bernatz S, Foldyna B, Lu MT, Fedorov A, Aerts HJWL.
[Coronary artery calcification assessment in National Lung Screening Trial CT images (DeepCAC2)](https://arxiv.org/abs/2603.24633).
2026. arXiv:2603.24633.
