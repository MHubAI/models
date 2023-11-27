"""
--------------------------------------------------------
Mhub / GC - Run Module for grt123 Lung Cancer Classifier
--------------------------------------------------------

--------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
--------------------------------------------------------
"""
import torch.cuda
from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, Meta

from typing import Dict
import json
from pathlib import Path

import torch

# Import the main module for the grt123 algorithm, which must be used for running the classification
import main


@ValueOutput.Name('lncancerprob')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Label('Lung Nodule cancer probability score.')
@ValueOutput.Type(float)
@ValueOutput.Description('The predicted cancer probability score for a single lung nodule detected by the algorithm')
class LNCancerProb(ValueOutput):
   pass


@ValueOutput.Name('clcancerprob')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Label('Case level cancer probability score.')
@ValueOutput.Type(float)
@ValueOutput.Description('The predicted cancer probability score for the whole case')
class CLCancerProb(ValueOutput):
    pass


# This method cleans the raw results from the grt123 algorithm output and only keeps the relevant details
def cleanup_json_report(data: Dict):
    for key in ["trainingset1", "trainingset2"]:
        del data["lungcad"][key]
    for key in ["patientuid", "studyuid"]:
        del data["imageinfo"][key]
    data["findings"] = [
        dict(
            id=f["id"],
            x=f["x"],
            y=f["y"],
            z=f["z"],
            probability=f["probability"],
            cancerprobability=f["cancerprobability"]
        )
        for f in data["findings"]
    ]


@IO.Config('n_preprocessing_workers', int, 6, the="number of preprocessing workers to use for the grt123 lung mask preprocessor")
class LungCancerClassifierRunner(Module):

    n_preprocessing_workers: int

    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the='input ct scan')
    @IO.Output('out_data', 'grt123_lung_cancer_findings.json', 'json:model=grt123LungCancerClassification', data='in_data', the='predicted nodules and lung cancer findings of the lung lobe')
    @IO.OutputData('clcancerprob', CLCancerProb, the='Case level probability score')
    @IO.OutputDatas('lncancerprobs', LNCancerProb, the='Individual lung nodule probability scores')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData, clcancerprob: CLCancerProb, lncancerprobs: LNCancerProb) -> None:
        # create temporary directories for the preprocessed data and the cropped bounding boxes
        tmp_path = Path(self.config.data.requestTempDir('grt123'))
        tmp_output_bbox_dir = tmp_path / "bbox"
        tmp_output_prep_dir = tmp_path / "prep"
        tmp_output_bbox_dir.mkdir(exist_ok=True, parents=True)
        tmp_output_prep_dir.mkdir(exist_ok=True, parents=True)

        # determine the number of GPUs we can use
        if torch.cuda.is_available():
            self.log("Running with a GPU", "NOTICE")
            n_gpu = 1
        else:
            self.log("Running on the CPU, might be slow...", "NOTICE")
            n_gpu = 0

        # apply grt123 algorithm
        results = main.main(
            skip_detect=False,
            skip_preprocessing=False,
            datapath=str(Path(in_data.abspath).parent),
            outputdir=str(tmp_path),
            output_bbox_dir=str(tmp_output_bbox_dir),
            output_prep_dir=str(tmp_output_prep_dir),
            n_gpu=n_gpu,
            n_worker_preprocessing=self.n_preprocessing_workers,
            data_filter=r".*.mha"
        )

        # retrieve classification results
        assert len(results) > 0, "LungCancerClassifierRunner - Always expects at least one output report"
        results_dict = results[0].to_json()
        cleanup_json_report(results_dict)

        # export to JSON (original json file)
        self.log(f"Writing classification results to {out_data.abspath}", "NOTICE")
        with open(out_data.abspath, "w") as f:
            json.dump(results_dict, f, indent=4)

        # set output value for case level cancer probability
        clcancerprob.value = results_dict["cancerinfo"]["casecancerprobability"]

        # set output values for nodule level cancer probabilities
        for finding in results_dict["findings"]:
            nodule_cancer_prob = LNCancerProb()
            nodule_cancer_prob.meta = Meta(id=finding['id'], x=finding['x'], y=finding['y'], z=finding['z'], )
            nodule_cancer_prob.description += f" (for nodule {finding['id']} at location ({finding['x']}, {finding['y']}, {finding['z']}))"
            nodule_cancer_prob.value = finding["cancerprobability"]
            lncancerprobs.add(nodule_cancer_prob)
