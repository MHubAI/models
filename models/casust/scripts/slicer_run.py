"""
-------------------------------------------------
MHub - 3D slicer specific run script for the 
       casust pipeline
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
Date:   13.04.2023
-------------------------------------------------
"""

import sys
from mhubio.core import Config, DataType, FileType, SEG
from mhubio.modules.importer.NrrdImporter import NrrdImporter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from mhubio.modules.runner.NNUnetRunner import NNUnetRunner
from models.casust.utils.CasustRunner import CasustRunner

# config
config = Config('/app/models/casust/config/slicer_config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 
config.debug = False

# load NRRD file (ct:nrrd)
NrrdImporter(config).execute()

# convert (ct:dicom -> ct:nifti)
NiftiConverter(config).execute()

# run the heart segmentation model
NNUnetRunner(config).execute()

# execute model (ct:nifti + heart:nifti -> seg:nifti)
CasustRunner(config).execute()

# organize data into output folder
organizer = DataOrganizer(config)
organizer.setTarget(DataType(FileType.NRRD, SEG), "/app/data/output_data/[d:roi].nii.gz")
organizer.execute()