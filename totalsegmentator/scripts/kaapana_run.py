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

from mhubio.Config import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.UnsortedDicomImporter import UnsortedInstanceImporter
from mhubio.modules.importer.DataSorter import DataSorter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from totalsegmentator.utils.TotalSegmentatorRunner import TotalSegmentatorRunner

# could be set via environment variables
# format: '/data/batch/series1/input/', '/data/batch/series1/output/'
input_dir = '/app/kaapana_volume/some/folder/input_data/'
output_dir = '/app/kaapana_volume/another/folder/output_data/'

# config
config = Config('/app/mhub/totalsegmentator/config/config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 
config.debug = True

# import 
importer = UnsortedInstanceImporter(config)
importer.setInputDir(input_dir)
importer.execute()

# sort
DataSorter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# execute model (ct:nifti -> seg:nifti)
TotalSegmentatorRunner(config).execute()

# convert (seg:nifti -> seg:dicomseg)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.DICOMSEG, SEG), os.path.join(output_dir, "TotalSegmentator.seg.dcm"))
organizer.execute()