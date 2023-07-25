"""
-------------------------------------------------------
MHub / DIAG - Run HookNet Lung Segmentation Model
              WSI Dicom input variant
-------------------------------------------------------

-------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from mhubio.modules.importer.DicomImporter import DicomImporter
from gc_tiger_lb2.utils.TigerLB2Runner import TigerLB2Runner
from gc_tiger_lb2.utils.PanImgConverters import TiffPanImgConverter

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/gc_tiger_lb2/config/default.yml')

# import (sm:dicom)
DicomImporter(config).execute()

# convert (sm:dicom -> sm:tiff)
TiffPanImgConverter(config).execute()

# execute model (sm:tiff -> json)
TigerLB2Runner(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux')).execute()
