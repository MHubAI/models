"""
-------------------------------------------------------------
MHub - PanImg Conversion Modules Dicom2Mha and WSI-Dicom2Tiff
-------------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""


from typing import Optional

from mhubio.modules.convert.DataConverter import DataConverter
from mhubio.core import Instance, InstanceData, DataType, FileType

import os
from pathlib import Path
import shutil

from panimg.exceptions import UnconsumedFilesException
from panimg.image_builders.dicom import image_builder_dicom
from panimg.image_builders.tiff import image_builder_tiff
from panimg.image_builders.metaio_nrrd import image_builder_nrrd

import SimpleITK


class MhaPanImgConverter(DataConverter):
    """
    Conversion module.
    Convert instance data from dicom or nrrd to mha.
    """

    def convert(self, instance: Instance) -> Optional[InstanceData]:

        # create a converted instance
        has_instance_dicom = instance.hasType(DataType(FileType.DICOM))
        has_instance_nrrd = instance.hasType(DataType(FileType.NRRD))

        assert has_instance_dicom or has_instance_nrrd, f"CONVERT ERROR: required datatype (dicom or nrrd) not available in instance {str(instance)}."

        # select input data, dicom has priority over nrrd
        input_data = instance.data.filter(DataType(FileType.DICOM) if has_instance_dicom else DataType(FileType.NRRD)).first()

        # out data
        mha_data = InstanceData("image.mha", DataType(FileType.MHA, input_data.type.meta))
        mha_data.instance = instance

        # paths
        inp_data_dir = Path(input_data.abspath)
        out_mha_file = Path(mha_data.abspath)

        # sanity check
        assert(inp_data_dir.is_dir())

        # DICOM CT to MHA conversion (if the file doesn't exist yet)
        if out_mha_file.is_file():
            print("CONVERT ERROR: File already exists: ", out_mha_file)
            return None
        else:
            # run conversion using panimg
            input_files = {f for f in inp_data_dir.glob(["*.nrrd", "*.dcm"][has_instance_dicom]) if f.is_file()}
            img_builder = image_builder_dicom if has_instance_dicom else image_builder_nrrd
            try:
                for result in img_builder(files=input_files):
                    sitk_image = result.image  # SimpleITK image
                    SimpleITK.WriteImage(sitk_image, str(out_mha_file))
            except UnconsumedFilesException as e:
                # e.file_errors is keyed with a Path to a file that could not be consumed,
                # with a list of all the errors found with loading it,
                # the user can then choose what to do with that information
                print("CONVERT ERROR: UnconsumedFilesException during PanImg conversion: ", e.file_errors)
                return None

            return mha_data


class TiffPanImgConverter(DataConverter):
    """
    Conversion module.
    Convert instance data from WSI-dicom to tiff.
    """

    def convert(self, instance: Instance) -> Optional[InstanceData]:

        # create a converted instance
        assert instance.hasType(DataType(FileType.DICOM)), f"CONVERT ERROR: required datatype (dicom) not available in instance {str(instance)}."
        dicom_data = instance.data.filter(DataType(FileType.DICOM)).first()

        # out data
        tiff_data = InstanceData("image.tiff", DataType(FileType.TIFF, dicom_data.type.meta))
        tiff_data.instance = instance

        # paths
        inp_dicom_dir = Path(dicom_data.abspath)
        out_tiff_file = Path(tiff_data.abspath)

        # sanity check
        assert(inp_dicom_dir.is_dir())

        # WSI-DICOM to TIFF conversion (if the file doesn't exist yet)
        if out_tiff_file.is_file():
            print("CONVERT ERROR: File already exists: ", out_tiff_file)
            return None
        else:
            # run conversion using panimg
            dcm_input_files = {f for f in inp_dicom_dir.glob("*.dcm") if f.is_file()}

            try:
                for result in image_builder_tiff(files=dcm_input_files):
                    tiff_image = result.file  # Path to the tiff file
                    shutil.move(str(tiff_image), str(out_tiff_file))
            except UnconsumedFilesException as e:
                # e.file_errors is keyed with a Path to a file that could not be consumed,
                # with a list of all the errors found with loading it,
                # the user can then choose what to do with that information
                print("CONVERT ERROR: UnconsumedFilesException during PanImg conversion: ", e.file_errors)
                return None

            return tiff_data
