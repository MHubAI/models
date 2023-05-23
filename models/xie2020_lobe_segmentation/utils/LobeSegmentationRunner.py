"""
------------------------------------------------------
Mhub / DIAG - Run Module for Xie2020 Lobe Segmentation
------------------------------------------------------

------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
------------------------------------------------------
"""

from typing import List

from mhubio.core import Instance, InstanceData, DataType, FileType, CT, SEG
from mhubio.modules.runner.ModelRunner import ModelRunner

import os
import numpy as np
import SimpleITK as sitk

from test import segment_lobe, segment_lobe_init


# TODO borrowed this from InstanceDataCollection, since it was not included in MHub base image yet...
def filterByDataType(pool: List[InstanceData], ref_type: DataType, confirmed_only: bool = True) -> List[InstanceData]:
    """
    Filter for instance data by a reference data type. Only instance data that match the file type and specified meta data of the reference type are returned. A datatype matches the reference type, if all metadata of the reference type is equal to the datatype. If a datatype contains additional meta data compared to the reference type (specialization) those additional keys are ignored.
    """

    # collect only instance data passing all checks (ftype, meta)
    matching_data: List[InstanceData] = []

    # iterate all instance data of this instance
    for data in pool:
        # check if data is confirmed
        if confirmed_only and not data.confirmed:
            continue

        # check file type, ignore other filetypes
        if ref_type.ftype is not FileType.NONE and not data.type.ftype == ref_type.ftype:
            continue

        # check if metadata is less general than ref_type's metadata
        if not data.type.meta <= ref_type.meta:
            continue

        # add instance data that passes all prior checks
        matching_data.append(data)

    return matching_data


class LobeSegmentationRunner(ModelRunner):
    def runModel(self, instance: Instance) -> None:

        # TODO input data originally was specified for MHA/MHD and could be extended for DICOM

        # data
        if isinstance(instance.data, list):
            data_instances = filterByDataType(pool=instance.data, ref_type=DataType(FileType.NRRD, CT), confirmed_only=False)
            assert any(data_instances)
            inp_data = data_instances[0]
        else:
            inp_data = instance.data.filter(DataType(FileType.NRRD, CT)).first()

        # read image
        self.v(f"Reading image from {inp_data.abspath}")
        img_itk = sitk.ReadImage(inp_data.abspath)
        img_np = sitk.GetArrayFromImage(img_itk)

        # apply lobe segmentation
        origin = img_itk.GetOrigin()[::-1]
        spacing = img_itk.GetSpacing()[::-1]
        direction = np.asarray(img_itk.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        meta_dict =  {"uid": os.path.basename(inp_data.abspath),
                     "size": img_np.shape,
                     "spacing": spacing,
                     "origin": origin,
                     "original_spacing": spacing,
                     "original_size": img_np.shape,
                     "direction": direction}
        handle = segment_lobe_init()
        seg_result_np = segment_lobe(handle, img_np, meta_dict)

        # store image
        out_file = os.path.join(instance.abspath, f'xie2020lobeseg.nrrd')
        self.v(f"Writing image to {out_file}")
        seg_itk = sitk.GetImageFromArray(seg_result_np)
        seg_itk.CopyInformation(img_itk)
        sitk.WriteImage(seg_itk, out_file)

        # meta
        meta = {
            "model": "Xie2020LobeSegmentation",
        }

        # create output data
        seg_data_type = DataType(FileType.NRRD, SEG + meta)           
        seg_data = InstanceData(out_file, type=seg_data_type)
        instance.addData(seg_data)
