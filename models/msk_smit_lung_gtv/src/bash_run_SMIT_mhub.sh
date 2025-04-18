#!/bin/bash
#
#
# Input arguments: 
# $1 data_dir
# $2 save_folder
# $3 load_weight_name

#Use SMIT
use_smit=1 #Use SMIT not SMIT+

#Data folder and there need a 'data.json' file in the folder 
data_dir="$1"

#Segmentation output folder 
save_folder="$2"

#Some configrations for the model, no need to change
#Trained weight 
load_weight_name="trained_weights/model_Mhub.pt"

#input_nifti="$4"
input_nifti=`ls data_dir/*.nii*`

a_min=-500
a_max=500
space_x=1.5
space_y=1.5
space_z=2.0
out_channels=2

python utils/gen_data_json.py $input_nifti

python run_segmentation.py \
    --roi_x 128 \
    --roi_y 128 \
    --roi_z 128 \
    --space_x $space_x \
    --space_y $space_y \
    --space_z $space_z \
    --data_dir $data_dir \
    --out_channels $out_channels \
    --load_weight_name $load_weight_name \
    --save_folder $save_folder \
    --a_min=$a_min \
    --a_max=$a_max \
    --use_smit $use_smit
