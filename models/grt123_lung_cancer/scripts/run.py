"""
------------------------------------------------------
MHub / GC - Run grt123 Lung Cancer Classifier locally
-----------------------------------------------------

-----------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------
"""

import sys
sys.path.append('.')

from mhubio.core import Config, DataType, FileType
from mhubio.modules.importer.DicomImporter import DicomImporter
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
from models.grt123_lung_cancer.utils.LungCancerClassifierRunner import LungCancerClassifierRunner
from models.grt123_lung_cancer.utils.PanImgConverters import MhaPanImgConverter

# clean-up
import shutil
shutil.rmtree("/app/data/sorted_data", ignore_errors=True)
shutil.rmtree("/app/tmp", ignore_errors=True)
shutil.rmtree("/app/data/output_data", ignore_errors=True)

# config
config = Config('/app/models/grt123_lung_cancer/config/config.yml')

# import (ct:dicom)
DicomImporter(config).execute()

# convert (ct:dicom -> ct:mha)
MhaPanImgConverter(config).execute()

# execute model (nnunet)
LungCancerClassifierRunner(config).execute()

# organize data into output folder
organizer = DataOrganizer(config, set_file_permissions=sys.platform.startswith('linux'))
organizer.setTarget(DataType(FileType.JSON), "/app/data/output_data/[i:sid]/grt123_lung_cancer_findings.json")
organizer.execute()
