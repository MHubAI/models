"""
------------------------------------------------------
MHub / DIAG - Slicer run for Xie2020 Lobe Segmentation
------------------------------------------------------

------------------------------------------------------
Author: Sil van de Leemput
Email:  s.vandeleemput@radboudumc.nl
------------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, SEG
from mhubio.modules.importer.NrrdImporter import NrrdImporter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.xie2020_lobe_segmentation.utils.LobeSegmentationRunner import LobeSegmentationRunner

# config
config = Config('/app/models/xie2020_lobe_segmentation/config/slicer_config.yml')
config.verbose = True

# load NRRD file (ct:nrrd)
NrrdImporter(config).execute()

# execute model (ct:nrrd -> seg:nrrd)
LobeSegmentationRunner(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.NRRD, SEG), "/app/data/output_data/xie2020lobeseg.nrrd")
organizer.execute()
