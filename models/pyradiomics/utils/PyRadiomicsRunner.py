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

from mhubio.core import Module, Instance, InstanceData, IO
import SimpleITK as sitk, numpy as np

class PyRadiomicsRunner(Module):

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
    @IO.Input('segmentation', 'nifti|nrrd:mod=seg', the='input segmentation')
    @IO.Output('results', '[d:roi].pyradiomics.csv', 'csv:features=pyradiomics', data='segmentation', the='output csv file with the results')
    def task(self, instance: Instance, image: InstanceData, segmentation: InstanceData, results: InstanceData) -> None:

        # imports
        # from radiomics import featureextractor, getTestCase

        # align nifti file 
        seg_file = self.align_seg_and_image(image.abspath, segmentation.abspath)

        # build pyradiomics cli command
        #  pyradiomics <path/to/image> <path/to/segmentation> -o results.csv -f csv --param <path/to/params.yaml>
        cmd = [
            'pyradiomics',
            image.abspath,
            seg_file,
            '-o', results.abspath,
            '-f', 'csv',
            '--param', '/app/models/pyradiomics/res/params.yml'
        ]

        self.log(cmd)

        # run pyradiomics
        self.subprocess(cmd)
        