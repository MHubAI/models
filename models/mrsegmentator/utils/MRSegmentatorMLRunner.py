"""
-------------------------------------------------
MHub - Run Module for MRSegmentator.
-------------------------------------------------

-------------------------------------------------
Author: Felix Dorfner
Email:  felix.dorfner@charite.de
-------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, FileType, CT, SEG, IO, DataTypeQuery
import os, subprocess
from segdb.classes.Segment import Segment
import shutil



# register custom segmentation before class definition
Segment.register("SPINE", name="Spine")

@IO.Config('use_fast_mode', bool, False, the="flag to set to run MRSegmentator in a faster mode")
class MRSegmentatorMLRunner(Module):

    use_fast_mode: bool

    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=ct', the="input whole body ct scan")
    @IO.Output('out_data', 'segmentations.nii.gz', 'nifti:mod=seg:model=MRSegmentator:roi=SPLEEN,RIGHT_KIDNEY,LEFT_KIDNEY,GALLBLADDER,LIVER,STOMACH,PANCREAS,RIGHT_ADRENAL_GLAND,LEFT_ADRENAL_GLAND,LEFT_LUNG,RIGHT_LUNG,HEART,AORTA,INFERIOR_VENA_CAVA,PORTAL_AND_SPLENIC_VEIN,LEFT_ILIAC_ARTERY,RIGHT_ILIAC_ARTERY,ESOPHAGUS,SMALL_INTESTINE,DUODENUM,COLON,URINARY_BLADDER,SPINE,SACRUM,LEFT_HIP,RIGHT_HIP,LEFT_FEMUR,RIGHT_FEMUR,LEFT_AUTOCHTHONOUS_BACK_MUSCLE,RIGHT_AUTOCHTHONOUS_BACK_MUSCLE,LEFT_ILIOPSOAS,RIGHT_ILIOPSOAS,LEFT_GLUTEUS_MAXIMUS,RIGHT_GLUTEUS_MAXIMUS,LEFT_GLUTEUS_MEDIUS,RIGHT_GLUTEUS_MEDIUS,LEFT_GLUTEUS_MINIMUS,RIGHT_GLUTEUS_MINIMUS', data='in_data', the="output segmentation mask containing all labels")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        
        tmp_dir = self.config.data.requestTempDir("mr_segmentator")

        raw_filename = os.path.basename(in_data.abspath)
        name_part = raw_filename.split('.nii')[0]
        extension = '.nii.gz'

        new_filename = f"{name_part}_seg{extension}"

        bash_command  = ["mrsegmentator"]
        bash_command += ["-i", in_data.abspath]
    
        bash_command += ["--outdir", tmp_dir]


        if self.use_fast_mode:
            self.v("Running MRSegmentator in lower memory footprint mode ('--split_level', 1)")
            self.v("Note: This increases runtime and possibly reduces segmentation performance.")
            bash_command += ["--split_level", "1"]
        else:
            self.v("Running MRSegmentator in default mode.")


        self.v(">> run: ", " ".join(bash_command))

        # run the model
        self.subprocess(bash_command, text=True)

        # copy data 
        shutil.copy(os.path.join(tmp_dir, new_filename), out_data.abspath)