"""
-------------------------------------------------
MHub - run pyradiomics
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
Date:   16.03.2024
-------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, IO, InstanceDataCollection
from mhubio.modules.organizer.DataOrganizer import DataOrganizer
import SimpleITK as sitk, numpy as np, pandas as pd, csv, os

@IO.Config('config_file', str, '/app/models/pyradiomics/res/params.yml', the='path to the pyradiomics parameter file')
@IO.Config('z_align_seg_and_image', bool, False, the='align segmentation and image in their z direction')
@IO.Config('mask_file_pattern', str, '[i:sid]/masks/[basename]', the='pattern to find the mask file')
@IO.Config('mask_file_column_name', str, 'Mask', the='column name in the pyradiomics batch processing file')
class PyRadiomicsRunner(Module):

    config_file: str
    z_align_seg_and_image: bool
    mask_file_pattern: str
    mask_file_column_name: str

    def align_seg_and_image(self, image_file: str, seg_file: str) -> str:
        
        img_vol = sitk.ReadImage(image_file)
        self.log('loaded image from ', image_file)

        seg_vol = sitk.ReadImage(seg_file)
        self.log('loaded seg from   ', seg_file, '\n')

        # 0) checks
        self.log('spacing:      ', img_vol.GetSpacing())
        assert np.all(np.isclose(img_vol.GetSpacing(), seg_vol.GetSpacing()))

        self.log('direction:    ', img_vol.GetDirection())
        assert np.all(np.isclose(img_vol.GetDirection(), seg_vol.GetDirection()))

        # ONLY if we have a size/ origin missmatch
        size_match = np.all(np.isclose(img_vol.GetSize(), seg_vol.GetSize()))
        origin_match = np.all(np.isclose(img_vol.GetOrigin(), seg_vol.GetOrigin()))

        if size_match and origin_match:
            self.log("---> all ok, nothing to align")
            return seg_file

        self.log()
        self.log('image size:   ', img_vol.GetSize())
        self.log('seg size:     ', seg_vol.GetSize())
        self.log('image origin: ', img_vol.GetOrigin())
        self.log('seg origin:   ', seg_vol.GetOrigin())
            
        # 1) convert image to matching origin and dimension
        identity = sitk.Transform(3, sitk.sitkIdentity)

        # Resample(input image, reference image, transformation, interpolator, default value)
        seg_vol_t = sitk.Resample(seg_vol, img_vol, identity, sitk.sitkNearestNeighbor, 0)

        # save temp file
        seg_file_t = seg_file + '.aligned.nrrd'
        sitk.WriteImage(seg_vol_t, seg_file_t)
        
        return seg_file_t

    @IO.Instance()
    @IO.Input('image', 'nifti|nrrd:mod=ct|mr',  the='input image ct or mr scan')
    @IO.Inputs('segmentation', 'nifti|nrrd:mod=seg', the='input segmentation')
    @IO.Output('results', 'pyradiomics.csv', 'csv:features=pyradiomics', data='image', the='output csv file with the results')
    def task(self, instance: Instance, image: InstanceData, segmentation: InstanceDataCollection, results: InstanceData) -> None:

        # request temp dir for pyradiomics batch processing file
        tmp_dir = self.config.data.requestTempDir('pyradiomics_config')
        pyr_bp_file = os.path.join(tmp_dir, 'pyradiomics_batch.csv')

        # mask column name
        # NOTE: "Mask" is the reserved column name that points to the mask file for pyradiomics.
        #       This path, however, only makes sense inside the docker container. To resolve this to outside masks, the
        #       user can specify a parttern (and use the same pattern in the DataOrganizer) and write this into a separate 
        #       column (as specified by mask_file_column_name). If mask_file_column_name is "Mask" we will write it as "_Mask"
        #       remporarily and then later replace "Mask" with "_Mask"
        if self.mask_file_column_name == "Mask":
            mask_column_name = "_Mask"
        else:
            mask_column_name = self.mask_file_column_name

        # prepare csv for pyradiomics batch processing
        # NOTE: we could allow custom columns that extract any metadata specified from the config but 
        #       for now we simply include the metadata string from each segmentation file
        with open(pyr_bp_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image', 'Mask', 'Label', 'Label_channel', 'MHub ROI', 'MHub Metadata', mask_column_name])
            
            for seg_file in segmentation:
                seg_rois = seg_file.type.meta['roi'].split(',')
            
                # optionally align nifti file 
                if self.z_align_seg_and_image:
                    seg_file_abspath = self.align_seg_and_image(image.abspath, seg_file.abspath)
                else:
                    seg_file_abspath = seg_file.abspath
                    
                # get mask file outside-mhub-path (from a pattern, which must be used in the DataOrganizer too then)
                seg_file_resolved_path = DataOrganizer.resolveTarget(self.mask_file_pattern, seg_file)
                    
                # write one row per roi into the pyradiomics batch processing csv
                for channel_id, seg_roi in enumerate(seg_rois):
                    writer.writerow([image.abspath, seg_file_abspath, 1, channel_id + 1, seg_roi, seg_file.type.meta, seg_file_resolved_path])

        # for debugging print the content of the pyr_bp_file 
        with open(pyr_bp_file, 'r') as f:
            self.log(f.read())

        # build pyradiomics cli command
        #  pyradiomics <path/to/image> <path/to/segmentation> -o results.csv -f csv --param <path/to/params.yaml>
        cmd = [
            'uv', 'run', '-p', '.venv38',
            'pyradiomics',
            pyr_bp_file,
            '-o', results.abspath,
            '-f', 'csv',
            '--param', self.config_file
        ]

        self.log(cmd)

        # run pyradiomics
        self.subprocess(cmd)
        
        # check if we need to replace the "Mask" column with the contents in the "_Mask" column
        if mask_column_name == "_Mask":
            df = pd.read_csv(results.abspath)
            df['Mask'] = df['_Mask']
            df.drop(columns=['_Mask'], inplace=True)
            df.to_csv(results.abspath, index=False)
