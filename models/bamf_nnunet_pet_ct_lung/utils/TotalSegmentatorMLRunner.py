"""
-------------------------------------------------
MHub - Run Module for TotalSegmentator.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, FileType, CT, SEG, IO, DataTypeQuery
import os, subprocess

mapping = {
    'spleen': 'SPLEEN',
    'kidney_right': 'RIGHT_KIDNEY',
    'kidney_left': 'LEFT_KIDNEY',
    'gallbladder': 'GALLBLADDER',
    'liver': 'LIVER',
    'stomach': 'STOMACH',
    'aorta': 'AORTA',
    'inferior_vena_cava': 'INFERIOR_VENA_CAVA',
    'portal_vein_and_splenic_vein': 'PORTAL_AND_SPLENIC_VEIN',
    'pancreas': 'PANCREAS',
    'adrenal_gland_right': 'RIGHT_ADRENAL_GLAND',
    'adrenal_gland_left': 'LEFT_ADRENAL_GLAND',
    'lung_upper_lobe_left': 'LEFT_UPPER_LUNG_LOBE',
    'lung_lower_lobe_left': 'LEFT_LOWER_LUNG_LOBE',
    'lung_upper_lobe_right': 'RIGHT_UPPER_LUNG_LOBE',
    'lung_middle_lobe_right': 'RIGHT_MIDDLE_LUNG_LOBE',
    'lung_lower_lobe_right': 'RIGHT_LOWER_LUNG_LOBE',
    'vertebrae_L5': 'VERTEBRAE_L5',
    'vertebrae_L4': 'VERTEBRAE_L4',
    'vertebrae_L3': 'VERTEBRAE_L3',
    'vertebrae_L2': 'VERTEBRAE_L2',
    'vertebrae_L1': 'VERTEBRAE_L1',
    'vertebrae_T12': 'VERTEBRAE_T12',
    'vertebrae_T11': 'VERTEBRAE_T11',
    'vertebrae_T10': 'VERTEBRAE_T10',
    'vertebrae_T9': 'VERTEBRAE_T9',
    'vertebrae_T8': 'VERTEBRAE_T8',
    'vertebrae_T7': 'VERTEBRAE_T7',
    'vertebrae_T6': 'VERTEBRAE_T6',
    'vertebrae_T5': 'VERTEBRAE_T5',
    'vertebrae_T4': 'VERTEBRAE_T4',
    'vertebrae_T3': 'VERTEBRAE_T3',
    'vertebrae_T2': 'VERTEBRAE_T2',
    'vertebrae_T1': 'VERTEBRAE_T1',
    'vertebrae_C7': 'VERTEBRAE_C7',
    'vertebrae_C6': 'VERTEBRAE_C6',
    'vertebrae_C5': 'VERTEBRAE_C5',
    'vertebrae_C4': 'VERTEBRAE_C4',
    'vertebrae_C3': 'VERTEBRAE_C3',
    'vertebrae_C2': 'VERTEBRAE_C2',
    'vertebrae_C1': 'VERTEBRAE_C1',
    'esophagus': 'ESOPHAGUS',
    'trachea': 'TRACHEA',
    'heart_myocardium': 'MYOCARDIUM',
    'heart_atrium_left': 'LEFT_ATRIUM',
    'heart_ventricle_left': 'LEFT_VENTRICLE',
    'heart_atrium_right': 'RIGHT_ATRIUM',
    'heart_ventricle_right': 'RIGHT_VENTRICLE',
    'pulmonary_artery': 'PULMONARY_ARTERY',
    'brain': 'BRAIN',
    'iliac_artery_left': 'LEFT_ILIAC_ARTERY',
    'iliac_artery_right': 'RIGHT_ILIAC_ARTERY',
    'iliac_vena_left': 'LEFT_ILIAC_VEIN',
    'iliac_vena_right': 'RIGHT_ILIAC_VEIN',
    'small_bowel': 'SMALL_INTESTINE',
    'duodenum': 'DUODENUM',
    'colon': 'COLON',
    'rib_left_1': 'LEFT_RIB_1',
    'rib_left_2': 'LEFT_RIB_2',
    'rib_left_3': 'LEFT_RIB_3',
    'rib_left_4': 'LEFT_RIB_4',
    'rib_left_5': 'LEFT_RIB_5',
    'rib_left_6': 'LEFT_RIB_6',
    'rib_left_7': 'LEFT_RIB_7',
    'rib_left_8': 'LEFT_RIB_8',
    'rib_left_9': 'LEFT_RIB_9',
    'rib_left_10': 'LEFT_RIB_10',
    'rib_left_11': 'LEFT_RIB_11',
    'rib_left_12': 'LEFT_RIB_12',
    'rib_right_1': 'RIGHT_RIB_1',
    'rib_right_2': 'RIGHT_RIB_2',
    'rib_right_3': 'RIGHT_RIB_3',
    'rib_right_4': 'RIGHT_RIB_4',
    'rib_right_5': 'RIGHT_RIB_5',
    'rib_right_6': 'RIGHT_RIB_6',
    'rib_right_7': 'RIGHT_RIB_7',
    'rib_right_8': 'RIGHT_RIB_8',
    'rib_right_9': 'RIGHT_RIB_9',
    'rib_right_10': 'RIGHT_RIB_10',
    'rib_right_11': 'RIGHT_RIB_11',
    'rib_right_12': 'RIGHT_RIB_12',
    'humerus_left': 'LEFT_HUMERUS',
    'humerus_right': 'RIGHT_HUMERUS',
    'scapula_left': 'LEFT_SCAPULA',
    'scapula_right': 'RIGHT_SCAPULA',
    'clavicula_left': 'LEFT_CLAVICLE',
    'clavicula_right': 'RIGHT_CLAVICLE',
    'femur_left': 'LEFT_FEMUR',
    'femur_right': 'RIGHT_FEMUR',
    'hip_left': 'LEFT_HIP',
    'hip_right': 'RIGHT_HIP',
    'sacrum': 'SACRUM',
    'face': 'FACE',
    'gluteus_maximus_left': 'LEFT_GLUTEUS_MAXIMUS',
    'gluteus_maximus_right': 'RIGHT_GLUTEUS_MAXIMUS',
    'gluteus_medius_left': 'LEFT_GLUTEUS_MEDIUS',
    'gluteus_medius_right': 'RIGHT_GLUTEUS_MEDIUS',
    'gluteus_minimus_left': 'LEFT_GLUTEUS_MINIMUS',
    'gluteus_minimus_right': 'RIGHT_GLUTEUS_MINIMUS',
    'autochthon_left': 'LEFT_AUTOCHTHONOUS_BACK_MUSCLE',
    'autochthon_right': 'RIGHT_AUTOCHTHONOUS_BACK_MUSCLE',
    'iliopsoas_left': 'LEFT_ILIOPSOAS',
    'iliopsoas_right': 'RIGHT_ILIOPSOAS',
    'urinary_bladder': 'URINARY_BLADDER'
}

# from totalsegmentator.map_to_binary import class_map
# ROI = ','.join(mapping[class_map['total'][ci]] for ci in range(1, 105))
ROI = 'SPLEEN,RIGHT_KIDNEY,LEFT_KIDNEY,GALLBLADDER,LIVER,STOMACH,AORTA,INFERIOR_VENA_CAVA,PORTAL_AND_SPLENIC_VEIN,PANCREAS,RIGHT_ADRENAL_GLAND,LEFT_ADRENAL_GLAND,LEFT_UPPER_LUNG_LOBE,LEFT_LOWER_LUNG_LOBE,RIGHT_UPPER_LUNG_LOBE,RIGHT_MIDDLE_LUNG_LOBE,RIGHT_LOWER_LUNG_LOBE,VERTEBRAE_L5,VERTEBRAE_L4,VERTEBRAE_L3,VERTEBRAE_L2,VERTEBRAE_L1,VERTEBRAE_T12,VERTEBRAE_T11,VERTEBRAE_T10,VERTEBRAE_T9,VERTEBRAE_T8,VERTEBRAE_T7,VERTEBRAE_T6,VERTEBRAE_T5,VERTEBRAE_T4,VERTEBRAE_T3,VERTEBRAE_T2,VERTEBRAE_T1,VERTEBRAE_C7,VERTEBRAE_C6,VERTEBRAE_C5,VERTEBRAE_C4,VERTEBRAE_C3,VERTEBRAE_C2,VERTEBRAE_C1,ESOPHAGUS,TRACHEA,MYOCARDIUM,LEFT_ATRIUM,LEFT_VENTRICLE,RIGHT_ATRIUM,RIGHT_VENTRICLE,PULMONARY_ARTERY,BRAIN,LEFT_ILIAC_ARTERY,RIGHT_ILIAC_ARTERY,LEFT_ILIAC_VEIN,RIGHT_ILIAC_VEIN,SMALL_INTESTINE,DUODENUM,COLON,LEFT_RIB_1,LEFT_RIB_2,LEFT_RIB_3,LEFT_RIB_4,LEFT_RIB_5,LEFT_RIB_6,LEFT_RIB_7,LEFT_RIB_8,LEFT_RIB_9,LEFT_RIB_10,LEFT_RIB_11,LEFT_RIB_12,RIGHT_RIB_1,RIGHT_RIB_2,RIGHT_RIB_3,RIGHT_RIB_4,RIGHT_RIB_5,RIGHT_RIB_6,RIGHT_RIB_7,RIGHT_RIB_8,RIGHT_RIB_9,RIGHT_RIB_10,RIGHT_RIB_11,RIGHT_RIB_12,LEFT_HUMERUS,RIGHT_HUMERUS,LEFT_SCAPULA,RIGHT_SCAPULA,LEFT_CLAVICLE,RIGHT_CLAVICLE,LEFT_FEMUR,RIGHT_FEMUR,LEFT_HIP,RIGHT_HIP,SACRUM,FACE,LEFT_GLUTEUS_MAXIMUS,RIGHT_GLUTEUS_MAXIMUS,LEFT_GLUTEUS_MEDIUS,RIGHT_GLUTEUS_MEDIUS,LEFT_GLUTEUS_MINIMUS,RIGHT_GLUTEUS_MINIMUS,LEFT_AUTOCHTHONOUS_BACK_MUSCLE,RIGHT_AUTOCHTHONOUS_BACK_MUSCLE,LEFT_ILIOPSOAS,RIGHT_ILIOPSOAS,URINARY_BLADDER'

@IO.ConfigInput('in_data', 'nifti:mod=ct', the="input data to run TotalSegmentator on")
@IO.Config('use_fast_mode', bool, True, the="flag to set to run TotalSegmentator in fast mode")
class TotalSegmentatorMLRunner(Module):

    use_fast_mode: bool

    @IO.Instance()
    @IO.Input('in_data', the="input whole body ct scan")
    @IO.Output('out_data', 'segmentations.nii.gz', 'nifti:mod=seg:model=TotalSegmentator:roi=SPLEEN,RIGHT_KIDNEY,LEFT_KIDNEY,GALLBLADDER,LIVER,STOMACH,AORTA,INFERIOR_VENA_CAVA,PORTAL_AND_SPLENIC_VEIN,PANCREAS,RIGHT_ADRENAL_GLAND,LEFT_ADRENAL_GLAND,LEFT_UPPER_LUNG_LOBE,LEFT_LOWER_LUNG_LOBE,RIGHT_UPPER_LUNG_LOBE,RIGHT_MIDDLE_LUNG_LOBE,RIGHT_LOWER_LUNG_LOBE,VERTEBRAE_L5,VERTEBRAE_L4,VERTEBRAE_L3,VERTEBRAE_L2,VERTEBRAE_L1,VERTEBRAE_T12,VERTEBRAE_T11,VERTEBRAE_T10,VERTEBRAE_T9,VERTEBRAE_T8,VERTEBRAE_T7,VERTEBRAE_T6,VERTEBRAE_T5,VERTEBRAE_T4,VERTEBRAE_T3,VERTEBRAE_T2,VERTEBRAE_T1,VERTEBRAE_C7,VERTEBRAE_C6,VERTEBRAE_C5,VERTEBRAE_C4,VERTEBRAE_C3,VERTEBRAE_C2,VERTEBRAE_C1,ESOPHAGUS,TRACHEA,MYOCARDIUM,LEFT_ATRIUM,LEFT_VENTRICLE,RIGHT_ATRIUM,RIGHT_VENTRICLE,PULMONARY_ARTERY,BRAIN,LEFT_ILIAC_ARTERY,RIGHT_ILIAC_ARTERY,LEFT_ILIAC_VEIN,RIGHT_ILIAC_VEIN,SMALL_INTESTINE,DUODENUM,COLON,LEFT_RIB_1,LEFT_RIB_2,LEFT_RIB_3,LEFT_RIB_4,LEFT_RIB_5,LEFT_RIB_6,LEFT_RIB_7,LEFT_RIB_8,LEFT_RIB_9,LEFT_RIB_10,LEFT_RIB_11,LEFT_RIB_12,RIGHT_RIB_1,RIGHT_RIB_2,RIGHT_RIB_3,RIGHT_RIB_4,RIGHT_RIB_5,RIGHT_RIB_6,RIGHT_RIB_7,RIGHT_RIB_8,RIGHT_RIB_9,RIGHT_RIB_10,RIGHT_RIB_11,RIGHT_RIB_12,LEFT_HUMERUS,RIGHT_HUMERUS,LEFT_SCAPULA,RIGHT_SCAPULA,LEFT_CLAVICLE,RIGHT_CLAVICLE,LEFT_FEMUR,RIGHT_FEMUR,LEFT_HIP,RIGHT_HIP,SACRUM,FACE,LEFT_GLUTEUS_MAXIMUS,RIGHT_GLUTEUS_MAXIMUS,LEFT_GLUTEUS_MEDIUS,RIGHT_GLUTEUS_MEDIUS,LEFT_GLUTEUS_MINIMUS,RIGHT_GLUTEUS_MINIMUS,LEFT_AUTOCHTHONOUS_BACK_MUSCLE,RIGHT_AUTOCHTHONOUS_BACK_MUSCLE,LEFT_ILIOPSOAS,RIGHT_ILIOPSOAS,URINARY_BLADDER', data='in_data', the="output segmentation mask containing all labels")
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData) -> None:
        
        # build command
        bash_command  = ["TotalSegmentator"]
        bash_command += ["-i", in_data.abspath]
    
        # multi-label output (one nifti file containing all labels instead of one nifti file per label)
        self.v("Generating multi-label output ('--ml')")
        bash_command += ["-o", out_data.abspath]
        bash_command += ["--ml"]

        # fast mode
        if self.use_fast_mode:
            self.v("Running TotalSegmentator in fast mode ('--fast', 3mm)")
            bash_command += ["--fast"]
        else:
            self.v("Running TotalSegmentator in default mode (1.5mm)")

        # TODO: remove 
        self.v(">> run: ", " ".join(bash_command))

        # run the model
        self.subprocess(bash_command, text=True)