"""
-------------------------------------------------
MHub - run the TS pipeline locally
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

import sys, os
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.totalsegmentator.utils.TotalSegmentatorRunner import TotalSegmentatorRunner

# clean
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/totalsegmentator/config/config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 

# import 
DicomImporter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# execute model (ct:nifti -> seg:nifti)
TotalSegmentatorRunner(config).execute()

# convert (seg:nifti -> seg:dicomseg)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.NIFTI, CT), "/app/data/output_data/[i:sid]/[path]")
organizer.setTarget(DataType(FileType.DICOMSEG, SEG), "/app/data/output_data/[i:sid]/TotalSegmentator.seg.dcm")
organizer.execute()