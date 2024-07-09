"""
-------------------------------------------------
MHub - NNU-Net Runner v2
       Runner for pre-trained nnunet v2 models. 
-------------------------------------------------

-------------------------------------------------
Author: Rahul Soni
Email:  rahul.soni@bamfhealth.com
-------------------------------------------------
"""


from typing import List, Optional
import os, subprocess, shutil
import SimpleITK as sitk, numpy as np
from mhubio.core import Module, Instance, InstanceData, DataType, FileType, IO


nnunet_dataset_name_regex = r"Dataset[0-9]{3}_[a-zA-Z0-9_]+"

@IO.ConfigInput('in_data', 'nifti:mod=mr', the="input data to run nnunet on")
@IO.Config('nnunet_dataset', str, None, the='nnunet dataset name')
@IO.Config('nnunet_config', str, None, the='nnunet model name (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)')
@IO.Config('folds', int, None, the='number of folds to run nnunet on')
@IO.Config('use_tta', bool, True, the='flag to enable test time augmentation')
@IO.Config('export_prob_maps', bool, False, the='flag to export probability maps')
@IO.Config('prob_map_segments', list, [], the='segment labels for probability maps')
@IO.Config('roi', str, None, the='roi or comma separated list of roi the nnunet segments')
class NNUnetRunnerV2(Module):

    nnunet_dataset: str
    nnunet_config: str
    input_data_type: DataType
    folds: int                          # TODO: support optional config attributes
    use_tta: bool
    export_prob_maps: bool
    prob_map_segments: list
    roi: str

    def export_prob_mask(self, nnunet_out_dir: str, ref_file: InstanceData, output_dtype: str = 'float32', structure_list: Optional[List[str]] = None):
        """
        Convert softmax probability maps to NRRD. For simplicity, the probability maps
        are converted by default to UInt8
        Arguments:
            model_output_folder : required - path to the folder where the inferred segmentation masks should be stored.
            ref_file            : required - InstanceData object of the generated segmentation mask used as reference file.
            output_dtype        : optional - output data type. Data type float16 is not supported by the NRRD standard,
                                            so the choice should be between uint8, uint16 or float32. 
            structure_list      : optional - list of the structures whose probability maps are stored in the 
                                            first channel of the `.npz` file (output from the nnU-Net pipeline
                                            when `export_prob_maps` is set to True). 
        Outputs:
            This function [...]
        """

        # initialize structure list
        if structure_list is None:
            if self.roi is not None:
                structure_list = self.roi.split(',')
            else:
                structure_list = []

        # sanity check user inputs
        assert(output_dtype in ["uint8", "uint16", "float32"])      

        # input file containing the raw information
        pred_softmax_fn = 'VOLUME_001.npz'
        pred_softmax_path = os.path.join(nnunet_out_dir, pred_softmax_fn)

        # parse NRRD file - we will make use of if to populate the header of the
        # NRRD mask we are going to get from the inferred segmentation mask
        sitk_ct = sitk.ReadImage(ref_file.abspath)

        # generate bundle for prob masks
        # TODO: we really have to create folders (or add this as an option that defaults to true) automatically
        prob_masks_bundle = ref_file.getDataBundle('prob_masks')
        if not os.path.isdir(prob_masks_bundle.abspath):
            os.mkdir(prob_masks_bundle.abspath)

        # load softmax probability maps
        pred_softmax_all = np.load(pred_softmax_path)["softmax"]

        # iterate all channels
        for channel in range(0, len(pred_softmax_all)):

            structure = structure_list[channel] if channel < len(structure_list) else f"structure_{channel}"
            pred_softmax_segmask = pred_softmax_all[channel].astype(dtype = np.float32)

            if output_dtype == "float32":
                # no rescale needed - the values will be between 0 and 1
                # set SITK image dtype to Float32
                sitk_dtype = sitk.sitkFloat32

            elif output_dtype == "uint8":
                # rescale between 0 and 255, quantize
                pred_softmax_segmask = (255*pred_softmax_segmask).astype(np.int32)
                # set SITK image dtype to UInt8
                sitk_dtype = sitk.sitkUInt8

            elif output_dtype == "uint16":
                # rescale between 0 and 65536
                pred_softmax_segmask = (65536*pred_softmax_segmask).astype(np.int32)
                # set SITK image dtype to UInt16
                sitk_dtype = sitk.sitkUInt16
            else:
                raise ValueError("Invalid output data type. Please choose between uint8, uint16 or float32.")
                
            pred_softmax_segmask_sitk = sitk.GetImageFromArray(pred_softmax_segmask)
            pred_softmax_segmask_sitk.CopyInformation(sitk_ct)
            pred_softmax_segmask_sitk = sitk.Cast(pred_softmax_segmask_sitk, sitk_dtype)

            # generate data
            prob_mask = InstanceData(f'{structure}.nrrd', DataType(FileType.NRRD, {'mod': 'prob_mask', 'structure': structure}), bundle=prob_masks_bundle)

            # export file
            writer = sitk.ImageFileWriter()
            writer.UseCompressionOn()
            writer.SetFileName(prob_mask.abspath)
            writer.Execute(pred_softmax_segmask_sitk)

            # check if the file was written
            if os.path.isfile(prob_mask.abspath):
                self.v(f" > prob mask for {structure} saved to {prob_mask.abspath}")
                prob_mask.confirm()

    @IO.Instance()
    @IO.Input("in_data", the="input data to run nnunet on")
    @IO.Output("out_data", 'VOLUME_001.nii.gz', 'nifti:mod=seg:model=nnunet', data='in_data', the="output data from nnunet")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        
        # get the nnunet model to run
        self.v("Running nnUNet_predict.")
        self.v(f" > dataset:     {self.nnunet_dataset}")
        self.v(f" > config:      {self.nnunet_config}")
        self.v(f" > input data:  {in_data.abspath}")
        self.v(f" > output data: {out_data.abspath}")

        # download weights if not found
        # NOTE: only for testing / debugging. For productiio always provide the weights in the Docker container.
        if not os.path.isdir(os.path.join(os.environ["WEIGHTS_FOLDER"], '')):
            print("Downloading nnUNet model weights...")
            bash_command = ["nnUNet_download_pretrained_model", self.nnunet_dataset]
            self.subprocess(bash_command, text=True)

        # bring input data in nnunet specific format
        # NOTE: only for nifti data as we hardcode the nnunet-formatted-filename (and extension) for now.
        assert in_data.type.ftype == FileType.NIFTI
        assert in_data.abspath.endswith('.nii.gz')
        inp_dir = self.config.data.requestTempDir(label="nnunet-model-inp")
        inp_file = f'VOLUME_001_0000.nii.gz'
        shutil.copyfile(in_data.abspath, os.path.join(inp_dir, inp_file))

        # define output folder (temp dir) and also override environment variable for nnunet
        out_dir = self.config.data.requestTempDir(label="nnunet-model-out")
        os.environ['nnUNet_results'] = out_dir

        # symlink nnunet input folder to the input data with python
        # create symlink in python
        # NOTE: this is a workaround for the nnunet bash script that expects the input data to be in a specific folder
        #       structure. This is not the case for the mhub data structure. So we create a symlink to the input data
        #       in the nnunet input folder structure.
        os.symlink(os.path.join(os.environ['WEIGHTS_FOLDER'], self.nnunet_dataset), os.path.join(out_dir, self.nnunet_dataset))
        
        # NOTE: instead of running from commandline this could also be done in a pythonic way:
        #       `nnUNet/nnunet/inference/predict.py` - but it would require
        #       to set manually all the arguments that the user is not intended
        #       to fiddle with; so stick with the bash executable

        # construct nnunet inference command
        bash_command  = ["nnUNetv2_predict"]
        bash_command += ["-i", str(inp_dir)]
        bash_command += ["-o", str(out_dir)]
        bash_command += ["-d", self.nnunet_dataset]
        bash_command += ["-c", self.nnunet_config]
        
        # add optional arguments
        if self.folds is not None:
            bash_command += ["-f", str(self.folds)]

        if not self.use_tta:
            bash_command += ["--disable_tta"]
        
        if self.export_prob_maps:
            bash_command += ["--save_probabilities"]

        self.v(f" > bash_command:     {bash_command}")
        # run command
        self.subprocess(bash_command, text=True)

        # output meta
        meta = {
            "model": "nnunet",
            "nnunet_dataset": self.nnunet_dataset,
            "nnunet_config": self.nnunet_config,
            "roi": self.roi
        }

        # get output data
        out_file = f'VOLUME_001.nii.gz'
        out_path = os.path.join(out_dir, out_file)

        # copy output data to instance
        shutil.copyfile(out_path, out_data.abspath)

        # export probabiliy maps if requested as dynamic data
        if self.export_prob_maps:
            self.export_prob_mask(str(out_dir), out_data, 'float32', self.prob_map_segments)

        # update meta dynamically
        out_data.type.meta += meta
