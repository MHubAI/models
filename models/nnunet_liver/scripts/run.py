"""
-------------------------------------------------
MHub - run the PP pipeline locally
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.UnsortedDicomImporter import UnsortedInstanceImporter
from mhubio.modules.importer.DataSorter import DataSorter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.runner.NNUnetRunner import NNUnetRunner
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer

# clean-up
import shutil
shutil.rmtree("/app/data/sorted", ignore_errors=True)
shutil.rmtree("/app/data/nifti", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/nnunet_liver/config/config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 
config.debug = False

# import
UnsortedInstanceImporter(config).execute()

# sort
DataSorter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# execute model (nnunet)
runner = NNUnetRunner(config)
runner.input_type = DataType(FileType.NIFTI, CT)
runner.nnunet_model = '3d_lowres'
runner.nnunet_task = 'Task003_Liver'
runner.execute()

# convert (seg:nifti -> seg:dcm)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.NIFTI, CT), "/app/data/output_data/image.nii.gz")
organizer.setTarget(DataType(FileType.NIFTI, SEG), "/app/data/output_data/liver.nii.gz")
organizer.setTarget(DataType(FileType.DICOMSEG, SEG), "/app/data/output_data/liver.seg.dcm")
organizer.execute()
