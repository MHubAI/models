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

from mhubio.core import Config, DataType, FileType, CT, SEG, Meta
from mhubio.modules.importer.FileStructureImporter import FileStructureImporter
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.importer.NrrdImporter import NrrdImporter
from mhubio.modules.convert.NiftiConverter import NiftiConverter
from mhubio.modules.runner.NNUnetRunner import NNUnetRunner
from mhubio.modules.convert.DsegConverter import DsegConverter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.gc_nnunet_pancreas import MhaImporter, GCNNUnetPancreasRunner, HEATMAP

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/gc_nnunet_pancreas/config/config.yml')
config.verbose = True  # TODO: define levels of verbosity and integrate consistently. 

# import (ct:dicom)
#DicomImporter(config).execute()

# import (ct:mha)
MhaImporter(config).execute()
#FileStructureImporter(config).execute()

# execute model (nnunet ct:mha -> (hm:mha, seg:mha))
GCNNUnetPancreasRunner(config).execute()

# convert (seg:nifti -> seg:dcm)
# DsegConverter(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.MHA, HEATMAP), "/app/data/output_data/[i:sid]/heatmap.mha")
organizer.setTarget(DataType(FileType.MHA, SEG), "/app/data/output_data/[i:sid]/pancreas.seg.mha")
#organizer.setTarget(DataType(FileType.DICOMSEG, SEG), "/app/data/output_data/[i:sid]/pancreas.seg.dcm")
organizer.execute()