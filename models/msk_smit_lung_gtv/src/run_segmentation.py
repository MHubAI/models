import SimpleITK as sitk
import os
import torch
import numpy as np
from edit_inference_utils import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
import argparse
from monai import transforms, data
from monai.handlers.utils import from_engine
from monai.data import decollate_batch, load_decathlon_datalist
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    SpatialPadd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)

from smit_models.smit import CONFIGS as CONFIGS_SMIT
import smit_models.smit as smit
from smit_models import smit_plus


from skimage.measure import label 
import scipy.ndimage.morphology as snm
import skimage


parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str,
                    help='pretrained checkpoint directory')

parser.add_argument('--data_dir', default='/scratch/input', type=str,
                    help='dataset directory')
parser.add_argument('--json_list',
                    default='/scratch/input/data.json',
                    type=str, help='dataset json file')

parser.add_argument('--pretrained_model_name', default='model.pt', type=str,
                    help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str,
                    help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=1 + 6, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=-140, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=260, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.8, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--load_weight_name', default='a', type=str, help='trained_weight')
parser.add_argument('--save_folder', default='a', type=str, help='output_folder')
parser.add_argument('--model_feature', default=96, type=int, help='model_imbeding_feature size')
parser.add_argument('--scale_intensity', action='store_true', help='')
parser.add_argument('--use_smit', default=0, type=int, help='use smit model')


# copy spacing and orientation info between sitk objects
def copy_info(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())

    return dst

#Additional functions to filter out the body 


# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh): # 
    mask_map = np.float32(img_2d > thresh)
    
    def getLargestCC(segmentation): # largest connected components
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask
	
	
def Get_body_wrapper(img, verbose = False, fg_thresh = 1e-4):
    
    fg_mask_vol = np.zeros(img.shape)
    for ii in range(fg_mask_vol.shape[0]):
        if verbose:
            print("doing {} slice".format(ii))
        _fgm = fg_mask2d(img[ii, ...], fg_thresh )
        

        fg_mask_vol[ii] = _fgm
        
    return fg_mask_vol
	

def main():
    args = parser.parse_args()

    img_folder = args.data_dir
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data_dir = args.data_dir
    datalist_json = os.path.join(args.data_dir, 'data.json')

    
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"],
                        pixdim=(args.space_x, args.space_y, args.space_z),
                        mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),

            EnsureTyped(keys=["image"]),
        ]
    )
    

    test_files = load_decathlon_datalist(datalist_json,
                                         True,
                                         "val",
                                         base_dir=data_dir)

    val_org_ds = data.Dataset(data=test_files, transform=val_org_transforms)
    val_org_loader = data.DataLoader(val_org_ds, batch_size=1, num_workers=4)

    print('val data size is ', len(val_org_loader))
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),

    ])

    args.test_mode = True
    val_loader = val_org_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_smit == 1:
        config = CONFIGS_SMIT['SMIT_config']
        model = smit.SMIT_3D_Seg(config,
                                 out_channels=args.out_channels,
                                 norm_name='instance')
    else:
        model = smit_plus.SMIT_Plus(out_channels=args.out_channels,
                                    in_channels=args.in_channels,
                                    norm_name='instance',
                                    feature_size=args.model_feature)

    model_dict = torch.load(args.load_weight_name)

    print('info: started to load weight: ', args.load_weight_name)
    print('info: model emb feature is : ', args.model_feature)
    model.load_state_dict(model_dict['state_dict'], strict=True)
    model.eval()
    model.to(device)
    print('info: Successfully loaded trained weights: ', args.load_weight_name)

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_inputs = val_data["image"].cuda()

            img_name = val_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]

            with autocast(enabled=True):
                val_data["pred"] = sliding_window_inference(val_inputs,
                                                            (args.roi_x, args.roi_y, args.roi_z),
                                                            4,
                                                            model,
                                                            overlap=args.infer_overlap)

            val_data = [post_transforms(i) for i in decollate_batch(val_data)]

            val_outputs = from_engine(["pred"])(val_data)

            val_outputs = val_outputs[0]

            seg_ori_size = val_outputs.numpy().astype(np.uint8)
            seg_ori_size = np.squeeze(seg_ori_size)

            pred_sv_name = os.path.join(save_folder, os.path.split(args.load_weight_name)[-1].replace('.pt', '') + '_' + img_name)

            print('info: start get the info')

            #Start to filter the body 
            cur_rd_path = os.path.join(img_folder, img_name)
            im_obj = sitk.ReadImage(cur_rd_path)
            img_3d_data=sitk.GetArrayFromImage(im_obj)
            threshold_= -150
            out_fg= Get_body_wrapper(img_3d_data, fg_thresh = threshold_)
            out_fg=np.transpose(out_fg, (2, 1, 0))
            seg_ori_size[out_fg==0]=0
            seg_ori_size = np.transpose(seg_ori_size, (2, 1, 0))
            out_fg_o = sitk.GetImageFromArray(seg_ori_size)
            seg_ori_size = copy_info(im_obj, out_fg_o)
            sitk.WriteImage(seg_ori_size, pred_sv_name)


if __name__ == '__main__':
    main()
