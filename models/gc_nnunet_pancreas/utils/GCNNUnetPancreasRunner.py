"""
-----------------------------------------------------------
GC / MHub - Run Module for the GC NNUnet Pancreas Algorithm
-----------------------------------------------------------

-----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, Meta, IO, ValueOutput

from pathlib import Path
import SimpleITK
import sys


CLI_PATH = Path(__file__).parent / "cli.py"


@ValueOutput.Name('prostate_cancer_likelihood')
@ValueOutput.Label('ProstateCancerLikelihood')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="likelihood"))
@ValueOutput.Type(float)
@ValueOutput.Description('Likelihood of case-level prostate cancer.')
class ProstateCancerLikelihood(ValueOutput):
    pass


class GCNNUnetPancreasRunner(Module):
    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the="input data")
    @IO.Output('heatmap', 'heatmap.mha', 'mha:mod=heatmap:model=GCNNUnetPancreas', data="in_data",
               the="raw heatmap of the pancreatic tumor likelihood (not masked with any pancreas segmentations).")
    @IO.Output('segmentation_raw', 'segmentation_raw.mha', 'mha:mod=seg:src=original:model=GCNNUnetPancreas:roi=VEIN,ARTERY,PANCREAS,PANCREATIC_DUCT,BILE_DUCT,PANCREAS+CYST,RENAL_VEIN', data="in_data",
               the="original segmentation of the pancreas, with the following classes: "
                   "0-background, 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, 7-renal vein")
    @IO.Output('segmentation', 'segmentation.mha', 'mha:mod=seg:src=cleaned:model=GCNNUnetPancreas:roi=VEIN,ARTERY,PANCREAS,PANCREATIC_DUCT,BILE_DUCT', data="in_data",
               the="cleaned segmentation of the pancreas, with the following classes: "
                   "0-background, 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct")
    @IO.OutputData('cancer_likelihood', ProstateCancerLikelihood, the='Case-level pancreatic tumor likelihood. This is equivalent to the maximum of the pancreatic tumor likelihood heatmap.')
    def task(self, instance: Instance, in_data: InstanceData, heatmap: InstanceData, segmentation_raw: InstanceData, segmentation: InstanceData, cancer_likelihood: ProstateCancerLikelihood, **kwargs) -> None:
        # Call the PDAC CLI
        # A CLI was used here to ensure the mhub framework properly captures the nnUNet stdout output
        cmd = [
            sys.executable,
            str(CLI_PATH),
            in_data.abspath,
            heatmap.abspath,
            segmentation_raw.abspath
        ]
        self.subprocess(cmd, text=True)

        # Remove cysts and renal vein classes from the original segmentation.
        # Insufficient training samples were present in the training data for these classes.
        # Hence, these classes should be omitted from the final output, since these are not
        # expected to produce reliable segmentations.
        self.clean_segementation(
            segmentation_in=segmentation_raw,
            segmentation_out=segmentation
        )

        # Extract case-level cancer likelihood
        cancer_likelihood.value = self.extract_case_level_cancer_likelihood(
            heatmap=heatmap
        )

    def clean_segementation(self, segmentation_in: InstanceData, segmentation_out: InstanceData):
        self.log("Cleaning output segmentation", level="NOTICE")
        seg_sitk = SimpleITK.ReadImage(segmentation_in.abspath)
        seg_numpy = SimpleITK.GetArrayFromImage(seg_sitk)
        seg_numpy[seg_numpy >= 6] = 0  # remove cysts and renal vein segmentation from original segmentation
        remapped_sitk = SimpleITK.GetImageFromArray(seg_numpy)
        remapped_sitk.CopyInformation(seg_sitk)
        SimpleITK.WriteImage(remapped_sitk, segmentation_out.abspath, True)

    def extract_case_level_cancer_likelihood(self, heatmap: InstanceData):
        self.log("Extracting case-level cancer likelihood", level="NOTICE")
        heatmap_sitk = SimpleITK.ReadImage(heatmap.abspath)
        f = SimpleITK.MinimumMaximumImageFilter()
        f.Execute(heatmap_sitk)
        cancer_likelihood = f.GetMaximum()
        assert 0.0 <= cancer_likelihood <= 1.0, "Cancer likelihood value must be in range [0.0, 1.0]"
        return cancer_likelihood
