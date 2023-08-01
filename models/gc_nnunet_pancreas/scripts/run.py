"""
---------------------------------------------------
GC / MHub - run the NNUnet GC pancreas segmentation
            pipeline
---------------------------------------------------

---------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
---------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.gc_nnunet_pancreas import GCNNUnetPancreasRunner
from models.gc_nnunet_pancreas.utils import MhaPanImgConverter

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/gc_nnunet_pancreas/config/config.yml')

# import (ct:dicom)
DicomImporter(config).execute()

# convert (ct:dicom -> ct:mha)
MhaPanImgConverter(config).execute()

# execute model (nnunet ct:mha -> (hm:mha, seg:mha))
GCNNUnetPancreasRunner(config).execute()

# convert (seg:mha -> seg:dcm)
DsegConverter(config).execute()

# organize data into output folder
DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux')).execute()
