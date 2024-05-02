from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData
import os, shutil
import SimpleITK as sitk
import numpy as np

class LungPostProcessor(Module):

    @IO.Instance()
    @IO.Input('in_rg_data', 'nifti:mod=seg:nnunet_task=Task775_CT_NSCLC_RG', the='input data from custom nnunet module')
    @IO.Input('in_nodules_data', 'nifti:mod=seg:nnunet_task=Task777_CT_Nodules', the='input data nnunet nodule')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf', data='in_rg_data',
               the="keep the two largest connected components of the segmentation and remove all other ones")
    def task(self, instance: Instance, in_rg_data: InstanceData, in_nodules_data: InstanceData, out_data: InstanceData):
        print('input nodules data: ' + in_nodules_data.abspath)
        print('store results in ' + out_data.abspath)
        in_nodules_img = sitk.ReadImage(in_nodules_data.abspath)
        in_nodules_d = sitk.GetArrayFromImage(sitk.ReadImage(in_nodules_data.abspath))
        in_rg_d = sitk.GetArrayFromImage(sitk.ReadImage(in_rg_data.abspath))
        seg_data = np.zeros(in_rg_d.shape)
        seg_data[in_rg_d == 1] = 1
        seg_data[in_nodules_data == 1] = 2
        seg_data[in_rg_d == 2] = 3
        seg_img = sitk.GetImageFromArray(in_rg_d)
        inp_dir = self.config.data.requestTempDir(label="lung-post-processor")
        inp_file = os.path.join(inp_dir, f'final.nii.gz')
        seg_img.CopyInformation(in_nodules_img)
        sitk.WriteImage(seg_img, inp_file)
        shutil.copyfile(inp_file, out_data.abspath)
