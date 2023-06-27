"""
-------------------------------------------------------
MHub / DIAG - Run HookNet Lung Segmentation Model
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
from tiger_lb2.utils.TigerLB2Runner import TigerLB2Runner
from tiger_lb2.utils.PanImgConverters import TiffPanImgConverter
from tiger_lb2.utils.TiffImporter import TiffImporter

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/tiger_lb2/config/config.yml')

# TODO could be WSI Dicom input alternatively

# import (wsi:dicom)
# DicomImporter(config).execute()

# convert (wsi:dicom -> wsi:tiff)
# TiffPanImgConverter(config).execute()

# import (wsi:tiff)
TiffImporter(config).execute()

# execute model (wsi:tiff -> json)
TigerLB2Runner(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.JSON), "/app/data/output_data/[i:sid]/tiger_lb2_tils_score.json")
organizer.execute()
