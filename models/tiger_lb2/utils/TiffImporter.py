"""
--------------------------------------
MHub / DIAG - Tiff importer
--------------------------------------

--------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
--------------------------------------
"""
import os
from typing import Optional
from pathlib import Path

from mhubio.modules.importer.DataImporter import IDEF, DataImporter, FileType
from mhubio.core import Meta, DirectoryChain


# TODO should be moved to mhubio/core/templates.py
WSI = Meta(mod="wsi")


class TiffImporter(DataImporter):
    def task(self) -> None:
        source_dir = self.c['source_dir']
        source_dc = DirectoryChain(path=source_dir, parent=self.config.data.dc)
        # input tiff file directory
        input_dir = source_dc.abspath
        self.v(f"{input_dir}")

        # add input tiff files as WSI images...
        self.setBasePath(input_dir)
        for input_tiff_file in Path(input_dir).glob("*.tif"):
            self.v(f"{input_tiff_file}")
            self.addTiffWSI(str(input_tiff_file), ref=None)

        # let the base module take over from here
        super().task()

    def addTiffWSI(self, path: str, ref: Optional[str] = None) -> None:
        _path  = self._resolvePath(path, ref)
        self.v("adding wsi in tiff format with resolved path: ", _path)
        assert os.path.isfile(_path) and _path.endswith('.tif'), f"Expect existing tiff file, '{_path}' was given instead."
        self._import_paths.append(IDEF(
            ref = ref,
            path = path,
            ftype = FileType.TIFF,
            meta = WSI
        ))

