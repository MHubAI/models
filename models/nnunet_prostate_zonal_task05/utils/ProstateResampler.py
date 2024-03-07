import os
import pyplastimatch as pypla

from mhubio.core import Module, Instance, DataType, InstanceData, FileType, IO

# for specific use case, resample ADC to match T2 (T2 is his 'sesired_grid' property value)
# TODO: add reference to colab notebook?
class ProstateResampler(Module):

    @IO.Instance()
    @IO.Input('in_data', 'nifti:part=ADC', the="ADC image")
    @IO.Input('fixed_data', 'nifti:part=T2', the="T2 image")
    @IO.Output('out_data', 'resampled.nii.gz', 'nifti:part=ADC:resampled_to=T2', data='in_data', the="ADC image resampled to T2")
    def task(self, instance: Instance, in_data: InstanceData, fixed_data: InstanceData, out_data: InstanceData):

        # log data
        log_data = InstanceData('_pypla.log', DataType(FileType.LOG, in_data.type.meta + {
            "log-origin": "plastimatch",
            "log-task": "resampling",
            "log-caller": "Resampler",
            "log-instance": str(instance)
        }), data=in_data, auto_increment=True)

        # process
        resample_args = {
            'input': in_data.abspath,
            'output': out_data.abspath,
            'fixed': fixed_data.abspath,
        }

        # TODO add log file
        pypla.resample(
            verbose=self.config.verbose,   
            path_to_log_file=log_data.abspath,
            **resample_args # type: ignore
        )