"""
------------------------------------------------------------------
Mhub / DIAG - Run Module for WSI Background Segmentation Algorithm
------------------------------------------------------------------

------------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------------------
"""
from typing import Optional

import tempfile

from mhubio.core import Instance, InstanceData, IO, Module

from pathlib import Path

@IO.Config(
    'input_spacing', float, 2.0,
    the="Desired input spacing to run the segmentation algorithm for. "
        "The closest level matching the spacing in the input Tiff image will be selected. "
        "Default is 2.0 micrometer."
)
@IO.Config(
    'output_spacing', Optional[float], None,
    the="Desired output spacing for the output segmentation. "
        "By default this matches the input_spacing."
)
@IO.Config(
    'spacing_tolerance', float, 0.25,
    the="Relative spacing tolerance with respect to the desired input_spacing. "
        "By default this is set to 25%."
)
class WSIBackgroundSegmentationRunner(Module):

    input_spacing: float
    output_spacing: Optional[float]
    spacing_tolerance: float

    CLI_SCRIPT_PATH = Path("/app") / "src" / "process.py"

    @IO.Instance()
    @IO.Input('in_data', 'tif|tiff:mod=sm', the='input whole slide image Tiff')
    @IO.Output('out_data', 'gc_wsi_bg_segmentation.tif', 'tiff:mod=seg:model=WSIBackgroundSegmentation', 'in_data', the='Background segmentation of the input WSI.')
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        output_spacing = self.output_spacing
        if self.output_spacing is None:
            output_spacing = self.input_spacing

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Execute the Tiger LB2 Algorithm through a Python subprocess and associated pipenv environment
            self.subprocess(
                [
                    "uv",
                    "run",
                    "-p", ".venv38",
                    "python",
                    str(self.CLI_SCRIPT_PATH),
                    in_data.abspath,
                    out_data.abspath,
                    "--work-dir",
                    tmp_dir,
                    "--input-spacing",
                    str(self.input_spacing),
                    "--output-spacing",
                    str(output_spacing),
                    "--spacing-tolerance",
                    str(self.spacing_tolerance)
                ]
            )

        # Validate that the required output was generated by the subprocess
        if not Path(out_data.abspath).is_file():
            raise FileNotFoundError(
                f"Couldn't find expected output file: `{out_data.abspath}`. "
                f"The subprocess `{self.CLI_SCRIPT_PATH}` did not generate the required output file."
            )
