import os
import numpy as np 
import SimpleITK as sitk

from src.test import segment_lobe, segment_lobe_init

def run(input_image_path: str, output_image_path: str):
  
  img_itk = sitk.ReadImage(input_image_path)
  img_np = sitk.GetArrayFromImage(img_itk)

  # apply lobe segmentation
  origin = img_itk.GetOrigin()[::-1]
  spacing = img_itk.GetSpacing()[::-1]
  direction = np.asarray(img_itk.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
  meta_dict =  {
      "uid": os.path.basename(input_image_path),
      "size": img_np.shape,
      "spacing": spacing,
      "origin": origin,
      "original_spacing": spacing,
      "original_size": img_np.shape,
      "direction": direction
  }

  handle = segment_lobe_init()
  seg_result_np = segment_lobe(handle, img_np, meta_dict)

  # store image
  print(f"Writing image to {output_image_path}")
  seg_itk = sitk.GetImageFromArray(seg_result_np)
  seg_itk.CopyInformation(img_itk)
  sitk.WriteImage(seg_itk, output_image_path)
  
# cli
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Run Xie2020 Lobe Segmentation')
  parser.add_argument('input_image_path', type=str, help='Path to input image')
  parser.add_argument('output_image_path', type=str, help='Path to output image')
  args = parser.parse_args()
  
  run(args.input_image_path, args.output_image_path)