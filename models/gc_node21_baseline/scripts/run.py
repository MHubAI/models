"""
----------------------------------------------------------
Mhub / DIAG - Run the GC Node21 baseline Algorithm locally
----------------------------------------------------------

-------------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-------------------------------------------------------------
"""

import sys, os
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, CT, SEG
from mhubio.modules.importer.DataSorter import DataSorter
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.gc_node21_baseline.utils.Node21BaselineRunner import Node21BaselineRunner

from models.gc_node21_baseline.utils import MhaPanImgConverter

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/gc_node21_baseline/config/default.yml')

# import (dicom)
DicomImporter(config).execute()

# convert (cr:dicom -> cr:mha)
MhaPanImgConverter(config).execute()

# execute model (cr:mha -> json)
Node21BaselineRunner(config).execute()

# organize data into output folder
DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux')).execute()
