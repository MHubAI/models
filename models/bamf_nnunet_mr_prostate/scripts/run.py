"""
-------------------------------------------------
MHub - run the NNUnet MR liver segmentation
       pipeline
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.runner.NNUnetRunner import NNUnetRunner
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer


# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/bamf_nnunet_mr_prostate/config/default.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 

# import (ct:dicom)
DicomImporter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# execute model (nnunet)
NNUnetRunner(config).execute()

# convert (seg:nifti -> seg:dcm)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.NIFTI, CT), "/app/data/output_data/[i:sid]/image.nii.gz")
organizer.setTarget(DataType(FileType.NIFTI, SEG), "/app/data/output_data/[i:sid]/liver.nii.gz")
organizer.setTarget(DataType(FileType.DICOMSEG, SEG), "/app/data/output_data/[i:sid]/liver.seg.dcm")
organizer.execute()