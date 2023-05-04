
"""
-------------------------------------------------
MHub - default run script for the 
       lungmask pipeline
-------------------------------------------------

-------------------------------------------------
Author: Dennis Bontempi
Email:  d.bontempi@maastrichtuniversity.nl
-------------------------------------------------
"""

import sys
from mhubio.core import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.UnsortedDicomImporter import UnsortedInstanceImporter
from mhubio.modules.importer.DataSorter import DataSorter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.lungmask.utils.LungMaskRunner import LungMaskRunner

# clean-up
import shutil
shutil.rmtree("/app/data/sorted", ignore_errors=True)
shutil.rmtree("/app/data/nifti", ignore_errors=True)

# config
config = Config('/app/models/lungmask/config/slicer_config.yml')

config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 
config.debug = True

# load NRRD file (ct:nrrd)
NrrdImporter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# execute model (ct:nifti + heart:nifti -> seg:nifti)
LungMaskRunner(config).execute()

# convert (seg:nifti -> seg:dicomseg)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config)
organizer.setTarget(DataType(FileType.NRRD, SEG), "/app/data/output_data/[d:roi].nii.gz")
organizer.execute()