"""
------------------------------------------------------
MHub / DIAG - Slicer run for Xie2020 Lobe Segmentation
------------------------------------------------------

------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
------------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType, SEG
from mhubio.modules.importer.NrrdImporter import NrrdImporter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.xie2020_lobe_segmentation.utils.LobeSegmentationRunner import LobeSegmentationRunner
from models.xie2020_lobe_segmentation.utils.PanImgConverters import MhaPanImgConverter

# config
config = Config('/app/models/xie2020_lobe_segmentation/config/slicer_config.yml')

# load NRRD file (ct:nrrd)
NrrdImporter(config).execute()

# convert (ct:nrrd -> ct:mha)
MhaPanImgConverter(config).execute()

# execute model (ct:mha -> seg:mha)
LobeSegmentationRunner(config).execute()

# TODO should probably be converted back to NRRD here...

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.MHA, SEG), "/app/data/output_data/xie2020lobeseg.mha")
organizer.execute()
