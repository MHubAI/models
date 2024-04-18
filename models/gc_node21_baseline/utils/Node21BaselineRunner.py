"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Node21 baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
import SimpleITK
import json
from pathlib import Path

from mhubio.core import Instance, InstanceData, IO, Module, Meta, ValueOutput, OutputDataCollection

# Import Node21 baseline nodule detection algorithm from the node21_detection_baseline repo
from process import Noduledetection


@ValueOutput.Name('noduleprob')
@ValueOutput.Label('Nodule probability score.')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Type(float)
@ValueOutput.Description('The predicted probability for a single lung nodule detected by the Node21Baseline algorithm.')
class NoduleProbability(ValueOutput):
   pass


@ValueOutput.Name('nodulebbox')
@ValueOutput.Label('Nodule 2D bounding box.')
@ValueOutput.Meta(Meta(format='json'))
@ValueOutput.Type(str)
@ValueOutput.Description('The predicted 2D bounding box for a single lung nodule detected by the Node21Baseline algorithm.')
class NoduleBoundingBox(ValueOutput):
   pass


class Node21BaselineRunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'mha|nrrd|nifti:mod=cr', the='input chest X-Ray')
    @IO.Output('out_data', 'nodules.json', 'json:model=Node21Baseline', 'in_data', the='Node21 baseline nodule prediction in JSON format')
    @IO.OutputDatas('nodule_probs', NoduleProbability)
    @IO.OutputDatas('nodule_bounding_boxes', NoduleBoundingBox)
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData, nodule_probs: OutputDataCollection, nodule_bounding_boxes: OutputDataCollection) -> None:
        # Read input image
        input_image = SimpleITK.ReadImage(in_data.abspath)

        # Run nodule detection algorithm on the input image and generate predictions
        tmp_path = Path("/app/tmp")
        predictions = Noduledetection(input_dir=tmp_path, output_dir=tmp_path).predict(input_image=input_image)

        # Export the predictions to a JSON file
        with open(out_data.abspath, "w") as f:
            json.dump(predictions, f, indent=4)

        # Export the relevant data
        for nodule_idx, box in enumerate(predictions["boxes"]):
            probability, corners = box["probability"], box["corners"]

            nodule_prob = NoduleProbability()
            nodule_prob.description += f" (for nodule {nodule_idx})"
            nodule_prob.meta = Meta(id=nodule_idx, min=0.0, max=1.0, type="probability")
            nodule_prob.value = probability

            nodule_bounding_box = NoduleBoundingBox()
            nodule_bounding_box.description += f" (for nodule {nodule_idx})"
            nodule_bounding_box.meta = Meta(id=nodule_idx, format="json")
            nodule_bounding_box.value = json.dumps(corners)

            nodule_probs.add(nodule_prob)
            nodule_bounding_boxes.add(nodule_bounding_box)
