from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData
import os, shutil
import SimpleITK as sitk
import numpy as np
from skimage import measure, filters


class LungPostProcessor(Module):

    def get_lung(self, ip_path):
        # get lung segmentation
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(ip_path))
        seg_data[seg_data > 0] = 1
        return seg_data

    def n_connected(self, img_data):
        img_data_mask = np.zeros(img_data.shape)
        img_data_mask[img_data >= 1] = 1
        img_filtered = np.zeros(img_data_mask.shape)
        blobs_labels = measure.label(img_data_mask, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if count >= 1 and count <= 2:
                print(key, value)
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data[img_filtered != 1] = 0
        return img_data

    def get_seg_img(self, lungs, nodules, ct_path):
        seg_data = np.zeros(lungs.shape)
        seg_data[lungs == 1] = 1
        seg_data[nodules == 1] = 2
        ref = sitk.ReadImage(ct_path)
        seg_img = sitk.GetImageFromArray(seg_data)
        seg_img.CopyInformation(ref)
        return seg_img

    @IO.Instance()
    @IO.Input('in_rg_data', 'nifti:mod=seg:nnunet_task=Task775_CT_NSCLC_RG', the='input data from rg nnunet module')
    @IO.Input('in_nodules_data', 'nifti:mod=seg:nnunet_task=Task777_CT_Nodules', the='input data from nodules nnunet nodule')
    @IO.Input('in_ct_data', 'nifti:mod=ct', the='input ct data')
    @IO.Output('out_data', 'bamf_processed.nii.gz', 'nifti:mod=seg:processor=bamf:roi=LUNG,LUNG+NODULE',
               data='in_rg_data', the="get lung and lung nodule segmentation file")
    def task(self, instance: Instance, in_rg_data: InstanceData, in_nodules_data: InstanceData,
             in_ct_data: InstanceData, out_data: InstanceData):

        print('input nodules data: ' + in_nodules_data.abspath)
        print('store results in ' + out_data.abspath)

        seg_data = self.get_lung(in_rg_data.abspath)
        lungs = self.n_connected(seg_data)

        nodules = self.get_lung(in_nodules_data.abspath)
        nodules[lungs == 0] = 0

        seg_img = self.get_seg_img(np.copy(lungs), np.copy(nodules), in_ct_data.abspath)
        process_dir = self.config.data.requestTempDir(label="nnunet-lung-processor")
        process_file = os.path.join(process_dir, f'final.nii.gz')
        sitk.WriteImage(
            seg_img,
            process_file,
        )
        shutil.copyfile(process_file, out_data.abspath)
