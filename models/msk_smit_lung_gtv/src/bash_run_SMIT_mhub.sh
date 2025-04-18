#!/bin/bash
use_smit=1
input_nifti="$1"
data_dir=`dirname "$input_nifti"`
ls -ld "$data_dir"
ls -l "$data_dir/data.json"
#Segmentation output folder
save_file=$2
save_folder=`dirname $save_file`
#Some configrations for the model, no need to change
#Trained weight
load_weight_name="trained_weights/model_Mhub.pt"
a_min=-500
a_max=500
space_x=1.5
space_y=1.5
space_z=2.0
out_channels=2
python utils/gen_data_json.py $input_nifti
python run_segmentation.py --roi_x 128 --roi_y 128 --roi_z 128 --space_x 1.5 --space_y $space_y --space_z $space_z --data_dir $data_dir --out_channels $out_channels --load_weight_name $load_weight_name --save_folder $save_folder --a_min=$a_min --a_max=$a_max --use_smit $use_smit
