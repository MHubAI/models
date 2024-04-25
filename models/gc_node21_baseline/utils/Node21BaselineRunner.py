"""
-------------------------------------------------------------
Mhub / DIAG - Run Module for the GC Node21 baseline Algorithm
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""
import json
import sys
from pathlib import Path

from mhubio.core import Instance, InstanceData, IO, Module, Meta, ValueOutput, OutputDataCollection


CLI_PATH = Path(__file__).parent.parent.absolute() / "src" / "cli.py"


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
        # build command (order matters!)
        cmd = [
            sys.executable,
            str(CLI_PATH),
            in_data.abspath,
            out_data.abspath
        ]

        # run the command as subprocess
        self.subprocess(cmd, text=True)

        # Confirm the expected output file was generated
        if not Path(out_data.abspath).is_file():
            raise FileNotFoundError(f"Node21BaseLineRunner - Could not find the expected "
                                    f"output file: {out_data.abspath}, something went wrong running the CLI.")

        # Read the predictions to a JSON file
        with open(out_data.abspath, "r") as f:
            predictions = json.load(f)

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
