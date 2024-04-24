"""
-------------------------------------------------
MHub - run the casust pipeline
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
Date:   13.04.2023
-------------------------------------------------
"""

import os, subprocess, shutil
from mhubio.core import Instance, InstanceData, IO
from mhubio.modules.runner.ModelRunner import ModelRunner

@IO.Config('test_time_augmentation', int, 0, the='number of iterations for test time augmentation. Set to 0 to disable test time augmentation.')
class CasustRunner(ModelRunner):
    
    test_time_augmentation: int

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=ct',  the='input ct scan')
    @IO.Input('heart', 'nifti:mod=seg', the='input heart segmentation')
    @IO.Output('al', 'Atrium_L.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=LEFT_ATRIUM', the='predicted segmentation of the left atrium')
    @IO.Output('ar', 'Atrium_R.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=RIGHT_ATRIUM', the='predicted segmentation of the right atrium')
    @IO.Output('vl', 'Ventricle_L.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=LEFT_VENTRICLE', the='predicted segmentation of the left ventricle')
    @IO.Output('vr', 'Ventricle_R.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=RIGHT_VENTRICLE', the='predicted segmentation of the right ventricle')
    @IO.Output('lad', 'Coronary_LAD.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=CORONARY_ARTERY_LAD', the='predicted segmentation of the left anterior descending coronary artery')
    @IO.Output('rca', 'Coronary_Atery_R.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=CORONARY_ARTERY_RIGHT', the='predicted segmentation of the right coronary artery')
    @IO.Output('cflx', 'Coronary_Atery_CFLX.nrrd', 'nrrd:mod=seg:model=CaSuSt:roi=CORONARY_ARTERY_CFLX', the='predicted segmentation of the circumflex branch of the coronary artery')
    def task(self, instance: Instance, image: InstanceData, heart: InstanceData, al: InstanceData, ar: InstanceData, vl: InstanceData, vr: InstanceData, lad: InstanceData, rca: InstanceData, cflx: InstanceData) -> None:

        # request temp out dir
        out_dir = self.config.data.requestTempDir('casust')

        # 1 prepare
        command = [ 'python3', 'models/casust/src/cli/prepare.py']
        command += ['--input_file', image.abspath]
        command += ['--hmask', heart.abspath]
        command += ['--output_dir', out_dir]
        command += ['--tta', str(self.test_time_augmentation)]

        self.v("> Running preparation (1/3): ", " ".join(command))
        self.subprocess(command)

        # 2 predict
        command =  [ 'python3', 'models/casust/src/cli/predict.py']
        command += ['--config', 'models/casust/src/config.json']
        command += ['--output_dir', out_dir]
        command += ['--device', 'cuda']
        #command += ['--roi', '']

        self.v("> Running prediction (2/3): ", " ".join(command))
        self.subprocess(command)

        # 3 finalize
        command =  [ 'python3', 'models/casust/src/cli/finalize.py']
        command += ['--input_file', image.abspath]
        command += ['--output_dir', out_dir]
        command += ['--config', 'models/casust/src/config.json']
        #command += ['--roi', '']

        self.v("> Running finalization (3/3): ", " ".join(command))
        self.subprocess(command)

        # copy over the output files
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Atrium_L.org.pred.nrrd'), al.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Atrium_R.org.pred.nrrd'), ar.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Ventricle_L.org.pred.nrrd'), vl.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Ventricle_R.org.pred.nrrd'), vr.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Coronary_LAD.org.pred.nrrd'), lad.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Coronary_Atery_R.org.pred.nrrd'), rca.abspath)
        shutil.copyfile(os.path.join(out_dir, 'nrrd', 'Coronary_Atery_CFLX.org.pred.nrrd'), cflx.abspath)