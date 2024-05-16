import os
import shutil
import SimpleITK as sitk
import numpy as np
from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData


class Registration(Module):

    @IO.Instance()
    @IO.Input('in_fixed_data', 'nifti:mod=pt', the='input pt data')
    @IO.Input('in_moving_data', 'nifti:mod=ct', the='input ct data')
    @IO.Output('out_data', 'VOL000_registered.nii.gz', 'nifti:mod=ct:registered=true', the="registered ct data")
    def task(self, instance: Instance, in_moving_data: InstanceData, in_fixed_data: InstanceData, out_data: InstanceData):
        """
        Perform registration
        """
        fixed = sitk.ReadImage(in_fixed_data.abspath, sitk.sitkFloat32)
        moving = sitk.ReadImage(in_moving_data.abspath, sitk.sitkFloat32)
        numberOfBins = 24
        samplingPercentage = 0.10
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfBins)
        R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 200)
        R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)

        def command_iteration(method):
            print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")

        R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

        outTx = R.Execute(fixed, moving)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(int(np.min(sitk.GetArrayFromImage(moving))))
        resampler.SetTransform(outTx)
        out = resampler.Execute(moving)
        tmp_dir = self.config.data.requestTempDir(label="registration-processor")
        output_path = os.path.join(tmp_dir, f'registered.nii.gz')
        out.CopyInformation(fixed)
        sitk.WriteImage(out, output_path)
        shutil.copyfile(output_path, out_data.abspath)