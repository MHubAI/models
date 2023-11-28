"""
-------------------------------------------------
MHub - Run Module for Platipy.
-------------------------------------------------

-------------------------------------------------
Author: Leonard Nürnberg
Email:  leonard.nuernberg@maastrichtuniversity.nl
-------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, IO
import subprocess

@IO.Config('path_to_config_file', str, None, the="path to the config file (if empty, platipy's default config is used)")
class PlatipyRunner(Module):

    path_to_config_file: str

    @IO.Instance()
    @IO.Input('in_data', 'nifti:mod=ct', the="input data")
    @IO.Output('hrt', 'Heart.nii.gz', 'nifti:mod=seg:model=Platipy:roi=HEART', bundle='platipy', in_signature=False, 
               the="segmentation of the heart")
    @IO.Output('pul', 'A_Pulmonary.nii.gz', 'nifti:mod=seg:model=Platipy:roi=PULMONARY_ARTERY', bundle='platipy', in_signature=False, 
               the="segmentation of the pulmonary artery")                         
    @IO.Output('aor', 'A_Aorta.nii.gz', 'nifti:mod=seg:model=Platipy:roi=AORTA', bundle='platipy', in_signature=False, 
               the="segmentation of the aorta")           
    @IO.Output('rv', 'Ventricle_R.nii.gz', 'nifti:mod=seg:model=Platipy:roi=RIGHT_VENTRICLE', bundle='platipy', in_signature=False,
               the="segmentation of the right ventricle")
    @IO.Output('lv', 'Ventricle_L.nii.gz', 'nifti:mod=seg:model=Platipy:roi=LEFT_VENTRICLE', bundle='platipy', in_signature=False,
               the="segmentation of the left ventricle")
    @IO.Output('cns', 'CN_Sinoatrial.nii.gz', 'nifti:mod=seg:model=Platipy:roi=SINOTRIAL_NODE', bundle='platipy', in_signature=False, 
               the="segmentation of the sinoatrial conducting node")
    @IO.Output('vtr', 'Valve_Tricuspid.nii.gz', 'nifti:mod=seg:model=Platipy:roi=TRICUSPID_VALVE', bundle='platipy', in_signature=False, 
               the="segmentation of the tricuspid valve")
    @IO.Output('vpu', 'Valve_Pulmonic.nii.gz', 'nifti:mod=seg:model=Platipy:roi=PULMONARY_VALVE', bundle='platipy', in_signature=False, 
               the="segmentation of the pulmonic valve")
    @IO.Output('al', 'Atrium_L.nii.gz', 'nifti:mod=seg:model=Platipy:roi=LEFT_ATRIUM', bundle='platipy', in_signature=False, 
               the="segmentation of the left atrium") 
    @IO.Output('ar', 'Atrium_R.nii.gz', 'nifti:mod=seg:model=Platipy:roi=RIGHT_ATRIUM', bundle='platipy', in_signature=False, 
               the="segmentation of the right atrium")
    @IO.Output('acl', 'A_Coronary_L.nii.gz', 'nifti:mod=seg:model=Platipy:roi=CORONARY_ARTERY_LEFT', bundle='platipy', in_signature=False, 
               the="segmentation of the left coronary artery")                       
    @IO.Output('acr', 'A_Coronary_R.nii.gz', 'nifti:mod=seg:model=Platipy:roi=CORONARY_ARTERY_RIGHT', bundle='platipy', in_signature=False,
               the="segmentation of the right coronary artery")
    @IO.Output('vmi', 'Valve_Mitral.nii.gz', 'nifti:mod=seg:model=Platipy:roi=MITRAL_VALVE', bundle='platipy', in_signature=False, 
               the="segmentation of the mitral valve")
    @IO.Output('vcs', 'V_Venacava_S.nii.gz', 'nifti:mod=seg:model=Platipy:roi=SUPERIOR_VENA_CAVA', bundle='platipy', in_signature=False, 
               the="segmentation of the superior vena cava")
    @IO.Output('cflx', 'A_Cflx.nii.gz', 'nifti:mod=seg:model=Platipy:roi=CORONARY_ARTERY_CFLX', bundle='platipy', in_signature=False,
               the="segmentation of the coronary artery cflx")  
    @IO.Output('vao', 'Valve_Aortic.nii.gz', 'nifti:mod=seg:model=Platipy:roi=AORTIC_VALVE', bundle='platipy', in_signature=False, 
               the="segmentation of the aortic valve")
    @IO.Output('cnav', 'CN_Atrioventricular.nii.gz', 'nifti:mod=seg:model=Platipy:roi=ATRIOVENTRICULAR_NODE', bundle='platipy', in_signature=False, 
               the="segmentation of the atrioventricular conducting node")                
    @IO.Output('lad', 'A_LAD.nii.gz', 'nifti:mod=seg:model=Platipy:roi=CORONARY_ARTERY_LAD', bundle='platipy', in_signature=False, 
               the="segmentation of the left anterior descending coronary artery") 
    def task(self, instance: Instance, in_data: InstanceData, **kwargs) -> None:

        # define model output bundle
        bundle = instance.getDataBundle("platipy")
        
        # build command
        bash_command  = ["platipy", "segmentation", "cardiac"]
        bash_command += ["-o", bundle.abspath]      # /path/to/output_folder
        bash_command += [in_data.abspath]           # /path/to/ct.nii.gz

        # use a custom config file if provided or run platipy with default config
        if self.path_to_config_file is not None:
            self.v("Running the hybrid cardiac segmentation with config file at: " + str(self.path_to_config_file))
            bash_command += ["--config", str(self.path_to_config_file)]
        else:
            self.v("Running the hybrid cardiac segmentation with default configuration.")

        # TODO: remove 
        self.v(">> run pp: ", " ".join(bash_command))

        # run the model
        self.subprocess(bash_command, text=True)