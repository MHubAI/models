"""
---------------------------------------------------------
Custom - Dicom to Nifti Conversion Module using SimpleITK
---------------------------------------------------------

-------------------------------------------------
Author: Jithendra Kumar
Email:  Jithendra.kumar@bamfhealth.com
-------------------------------------------------

"""

from enum import Enum
from typing import List, Dict, Any

from mhubio.core import Module, Instance, InstanceDataCollection, InstanceData, DataType, FileType
from mhubio.core.IO import IO

import pydicom
import shutil
import os, subprocess
import pyplastimatch as pypla # type: ignore
import SimpleITK as sitk
from pathlib import Path


@IO.ConfigInput('in_datas', 'dicom:mod=ct|pt', the="target data that will be converted to nifti")
@IO.Config('allow_multi_input', bool, False, the='allow multiple input files')
#@IO.Config('targets', List[DataType], ['dicom:mod=ct', 'nrrd:mod=ct'], factory=IO.F.list(DataType.fromString), the='target data types to convert to nifti')
@IO.Config('bundle_name', str, 'nifti', the="bundle name converted data will be added to")
@IO.Config('converted_file_name', str, '[filename].nii.gz', the='name of the converted file')
@IO.Config('overwrite_existing_file', bool, False, the='overwrite existing file if it exists')
#@IO.Config('wrap_output', bool, False, the='Wrap output in bundles. Required, if multiple input data is allowed that is not yet separated into distinct bundles.')
class SitkNiftiConverter(Module):
    """
    Conversion module. 
    Convert instance data from dicom
    """

    allow_multi_input: bool
    bundle_name: str                    # TODO optional type declaration
    converted_file_name: str
    overwrite_existing_file: bool
    #wrap_output: bool

    @IO.Instance()
    #@IO.Inputs('in_datas', IO.C('targets'), the="data to be converted")
    @IO.Inputs('in_datas', the="data to be converted")
    @IO.Outputs('out_datas', path=IO.C('converted_file_name'), dtype='nifti:converter=sitk', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="converted data")
    @IO.Outputs('log_datas', path='[basename].pmconv.log', dtype='log:log-task=conversion', data='in_datas', bundle=IO.C('bundle_name'), auto_increment=True, the="log generated by conversion engine")
    def task(self, instance: Instance, in_datas: InstanceDataCollection, out_datas: InstanceDataCollection, log_datas: InstanceDataCollection, **kwargs) -> None:

        # some sanity checks
        assert isinstance(in_datas, InstanceDataCollection)
        assert isinstance(out_datas, InstanceDataCollection)
        assert len(in_datas) == len(out_datas)
        assert len(in_datas) == len(log_datas)

        print("debug NiftiConverter 1 len(in_datas)",len(in_datas))
        print("debug NiftiConverter 2 len(out_datas)",len(out_datas))
        # filtered collection must not be empty
        if len(in_datas) == 0:
            print(f"CONVERT ERROR: no data found in instance {str(instance)}.")
            return None

        # check if multi file conversion is enables
        if not self.allow_multi_input and len(in_datas) > 1:
            print("WARNING: found more than one matching file but multi file conversion is disabled. Only the first file will be converted.")
            in_datas = InstanceDataCollection([in_datas.first()])

        # conversion step
        for i, in_data in enumerate(in_datas):
            out_data = out_datas.get(i)
            log_data = log_datas.get(i)

            # check if output data already exists
            if os.path.isfile(out_data.abspath) and not self.overwrite_existing_file:
                print("CONVERT ERROR: File already exists: ", out_data.abspath)
                continue

            # check datatype 
            if in_data.type.ftype == FileType.DICOM:
                files = []
                dcm_dir = Path(in_data.abspath)
                for f in dcm_dir.glob("*.dcm"):
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    slicer_loc = ds.SliceLocation if hasattr(ds, "SlicerLocation") else 0
                    files.append((slicer_loc, f))
                slices = sorted(files, key=lambda s: s[0])
                ordered_files = [x[1] for x in slices]

                o_tmp_dir = self.config.data.requestTempDir(label="convert-processor")
                ptmp_dir = Path(o_tmp_dir)
                for i, f in enumerate(ordered_files):
                    shutil.copy(f, ptmp_dir / f"{i}.dcm")
                # load in with SimpleITK
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(ptmp_dir))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # save as nifti
                sitk.WriteImage(image, out_data.abspath, useCompression=True, compressionLevel=9)

            else:
                raise ValueError(f"CONVERT ERROR: unsupported file type {in_data.type.ftype}.")