"""
---------------------------------------------------
MHub / DIAG - Run Xie2020 Lobe Segmentation locally
---------------------------------------------------

---------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
---------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, SEG
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.convert.NrrdConverter import NrrdConverter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.xie2020_lobe_segmentation.utils.LobeSegmentationRunner import LobeSegmentationRunner

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/xie2020_lobe_segmentation/config/config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 

# import (ct:dicom)
DicomImporter(config).execute()

# convert (ct:dicom -> ct:nrrd)
NrrdConverter(config).execute()

# execute model (nnunet)
LobeSegmentationRunner(config).execute()

# convert (seg:nifti -> seg:dicomseg)
DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
#organizer.setTarget(DataType(FileType.NRRD, SEG), "/app/data/output_data/[i:sid]/lunglobes_rtsunet.nrrd")
organizer.setTarget(DataType(FileType.DICOMSEG, SEG), "/app/data/output_data/[i:sid]/lunglobes_rtsunet.seg.dcm")
organizer.execute()
