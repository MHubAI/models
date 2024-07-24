from mhubio.core import IO
from mhubio.core import Module, Instance, InstanceData
import os, shutil
from pathlib import Path
import tempfile
import subprocess

# TODO remove this code once the segdb is updated with the below code
import yaml
import segdb

custom_seg_config = """
segdb:
    triplets:
        T_EDEMA_MABNORMALITY:
            code: 79654002
            meaning: Edema
            scheme_designator: SCT
        T_NECROSIS_MABNORMALITY:
            code: 6574001
            meaning: Necrosis
            scheme_designator: SCT
        T_ENHANCING_MABNORMALITY:
            code: C113842
            meaning: Enhancing Lesion
            scheme_designator: NCIt
    segments:
        EDEMA:
            name: Edema
            category: C_MORPHOLOGICALLY_ABNORMAL_STRUCTURE
            type: T_EDEMA_MABNORMALITY
            color: [140, 224, 228]
        NECROSIS:
            name: Necrosis
            category: C_MORPHOLOGICALLY_ABNORMAL_STRUCTURE
            type: T_EDEMA_MABNORMALITY
            color: [216, 191, 216]
        ENHANCING:
            name: Enhancing Lesion
            category: C_MORPHOLOGICALLY_ABNORMAL_STRUCTURE
            type: T_ENHANCING_MABNORMALITY
            color: [128, 174, 128]                        
"""
parsed_config = yaml.safe_load(custom_seg_config)

if 'segdb' in parsed_config:
    if 'segments' in parsed_config['segdb'] and isinstance(parsed_config['segdb']['segments'], dict):
        from segdb.classes.Segment import Segment

        for seg_id, seg_data in parsed_config['segdb']['segments'].items():
            print("added segment", seg_id, seg_data)
            Segment.register(seg_id, **seg_data)

    if 'triplets' in parsed_config['segdb'] and isinstance(parsed_config['segdb']['triplets'], dict):
        from segdb.classes.Triplet import Triplet

        for trp_id, trp_data in parsed_config['segdb']['triplets'].items():
            print("added triplet", trp_id, trp_data)
            Triplet.register(trp_id, overwrite=True, **trp_data)


class BrainProcessor(Module):

    def _setup(self, t1: str, t1c: str, t2: str, flair: str, seg_dir: str):

        self.t1 = Path(t1)
        self.t2 = Path(t2)
        self.t1c = Path(t1c)
        self.flair = Path(flair)
        temp_folder = tempfile.mkdtemp()
        self.output_dir = Path(temp_folder)
        self.seg_dir = Path(seg_dir)
        self._find_ants()
        self.find_fsl()

        self.atlas: list[tuple[Path, Path]] = []
        self.sri_24_atlas = Path(__file__).parent / "templates/T1_brain.nii"

    def _find_ants(self, default_path="/usr/local/ants/"):
        cmd = shutil.which("N4BiasFieldCorrection")
        if cmd:
            return

        if (Path(default_path) / "bin" / "N4BiasFieldCorrection").exists():
            # add to PATH
            os.environ["PATH"] += os.pathsep + str(
                (Path(default_path) / "bin").resolve()
            )
            return
        raise FileNotFoundError("ANTs installation not found")

    def find_fsl(self, default_path="/usr/local/fsl/"):
        if "FSLDIR" not in os.environ:
            os.environ["FSLDIR"] = "/usr/local/fsl"
        # The fsl.sh shell setup script adds the FSL binaries to the PATH
        self.fsl_path = Path(os.environ["FSLDIR"])
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        os.environ["FSLTCLSH"] = f"{self.fsl_path}/bin/fsltclsh"
        os.environ["FSLWISH"] = f"${self.fsl_path}/bin/fslwish"
        os.environ["FSL_SKIP_GLOBA"] = "0"
        os.environ["FSLMULTIFILEQUIT"] = "TRUE"
        assert Path(self.fsl_path / "bin/flirt").exists(), "FSL installation not found"

    # step 1: Reorient images to RAI (RIGHT, ANTERIOR, POSTERIOR)
    def reorient(self, input_image: Path) -> Path:
        assert input_image.exists(), f"{input_image} not found"
        # step 1 : reorient
        # Reorient to RAI
        output_image = (
                self.output_dir / f"{input_image.name.replace('.nii.gz', '')}_rai.nii.gz"
        )
        reorient_command = [
            str(self.fsl_path / "bin" / "fslreorient2std"),
            str(input_image),
            str(output_image),
            "-s",
        ]
        self.v("reorienting....", reorient_command)
        subprocess.run(reorient_command, check=True)
        return output_image

    # step 2: N4 Bias correction
    def n4bias_correction(self, input_image: Path) -> Path:
        assert input_image.exists(), f"{input_image} not found"
        output_image = (
                self.output_dir / f"{input_image.name.replace('.nii.gz', '')}_n4.nii.gz"
        )
        # N4 Bias Field Correction command
        n4_correction_command = [
            "N4BiasFieldCorrection",
            "-d",
            "3",  # 3D image
            "-i",
            str(input_image),
            "-o",
            str(output_image),
        ]
        self.v("running N4BiasFieldCorrection....", n4_correction_command)
        # Run the N4 Bias Field Correction
        subprocess.run(n4_correction_command, check=True, capture_output=True)
        return output_image

    # step 3: Registration
    # Rigid registration using FLIRT
    def run_flirt(self, input_image: Path, reference_image: Path, ref_code="_registered_to_t1c"):
        assert input_image.exists(), f"{input_image} not found"
        assert reference_image.exists(), f"{reference_image} not found"
        # if input_image == reference_image:
        #     return input_image, input_image
        output_image = (
                self.output_dir
                / f"{input_image.name.replace('.nii.gz', '')}{ref_code}.nii.gz"
        )
        transformation_matrix = (
                self.output_dir / f"{input_image.name.replace('.nii.gz', '')}{ref_code}.mat"
        )
        cmd = [
            "flirt",
            "-in",
            str(input_image),
            "-ref",
            str(reference_image),
            "-out",
            str(output_image),
            "-omat",
            str(transformation_matrix),  # Save the transformation matrix
            "-dof",
            "6",  # 6 degrees of freedom for rigid registration
        ]
        self.v("running FLIRT....", cmd)
        subprocess.run(cmd, check=True)
        return output_image, transformation_matrix

    def standard_space_registration(self, input_image: Path):
        assert input_image.exists(), f"{input_image} not found"
        input_image_list = [img[0] for img in self.registered]
        mat = [img[1] for img in self.registered]

        t1 = input_image_list[1]
        output_image = (
                self.output_dir / f"{input_image.name.replace('.nii.gz', '')}_warpped.nii.gz"
        )
        transformation_matrix = (
                self.output_dir / f"{input_image.name.replace('.nii.gz', '')}_warpped.mat"
        )

        cmd = [
            "flirt",
            "-in",
            str(input_image),
            "-ref",
            str(self.sri_24_atlas),
            "-out",
            str(output_image),
            "-omat",
            str(transformation_matrix),  # Save the transformation matrix
            "-dof",
            "12",  # 6 degrees of freedom for rigid registration
        ]
        self.v("running FLIRT spr....", cmd)
        subprocess.run(cmd, check=True)
        # for image, tmat in self.registered:
        #     output_image =self.output_dir / str(Path(input_image.stem).stem+f"_warpped_nl.nii.gz")
        #     cmd = [
        #         self.flirt_cmd,
        #         '--in', image,
        #         '--aff', tmat,
        #         '--iout', output_image,
        #     ]
        return output_image, transformation_matrix

    def skullstripping(self, input_image: Path, mask=False) -> Path:
        assert input_image.exists(), f"{input_image} not found"
        output_image = (
                self.output_dir
                / f"{input_image.name.replace('.nii.gz', '')}_skull_stripped.nii.gz"
        )
        # synthstrip_docker = Path("synthstrip-docker")
        # assert synthstrip_docker.exists(), "synthstrip-docker not found"
        synth_cmd = [
            "mri_synthstrip",
            "-i",
            str(input_image),
            "-o",
            str(output_image),
        ]
        subprocess.run(synth_cmd, check=True)
        return output_image

    def force_symlink(self, file1, file2):
        if file2.exists():
            os.remove(file2)
            os.symlink(file1, file2)
        else:
            os.symlink(file1, file2)

    def infer_brain_tumor(self, config="3d_fullres", predict=True):
        temp_folder = tempfile.mkdtemp()
        # out_folder = tempfile.mkdtemp()
        # seg_output_dir = Path(self.seg_dir)
        studyid = []
        for cindex, image_path in enumerate(self.atlas):
            row_folder = Path(temp_folder)
            studyid.append(Path(image_path[0]).parts[-1])
            self.sub_id = self.t1c.parts[-2]
            # Copy images from the DataFrame to the temporary folder
            if image_path[0]:
                self.force_symlink(
                    image_path[0],
                    Path(
                        row_folder / str(f"{self.sub_id}_{str(cindex).zfill(4)}.nii.gz")
                    ),
                )
        self.v("predicting....", predict)
        if True:
            cmd = f"nnUNetv2_predict -i {str(row_folder)} -o {self.output_dir} -d 002 -c {config}"
            self.v("running nnUNet inference....", cmd)
            os.system(cmd)
            self.v("prediction done....")
            # shutil.rmtree(row_folder)
        self.segmentation = self.output_dir / str(f"{self.sub_id}.nii.gz")
        return studyid

    def forward_preprocess(self, predict=True):
        self.mr_contrasts = [self.t1, self.t1c, self.t2, self.flair]
        self.reoriented = [
            self.reorient(input_image) for input_image in self.mr_contrasts
        ]
        self.n4_corrected = [self.n4bias_correction(image) for image in self.reoriented]
        self.registered = [
            self.run_flirt(image, self.n4_corrected[0]) for image in self.n4_corrected
        ]
        self.brain_mask = self.skullstripping(self.registered[1][0])
        self.skull_stripped = [
            self.skullstripping(image[0], self.brain_mask) for image in self.registered
        ]
        # self.atlas =[self.run_flirt(image[0], self.sri_24_atlas, "_registered_to_atlas") for image in self.registered] # TODO: need modification
        self.atlas = [
            self.standard_space_registration(image) for image in self.skull_stripped
        ]
        self.v("running brain tumor segmentation....", self.atlas)
        self.studyuids = self.infer_brain_tumor(config="3d_fullres", predict=predict)
        if predict:
            seg_copy = self.seg_dir / "std_space/"
            seg_copy.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.segmentation, seg_copy)
            self.v(f"copied to {seg_copy}")

    def inverse_registration_with_transformation(
            self,
            input_image: Path,
            transformation_matrix: Path,
            ref_image: Path,
            study,
            appendix="_back_to_patient.nii.gz",
    ) -> Path:
        # Output image path in patient space
        output_image = self.output_dir / str(study + appendix)
        reverse_transformation_matrix = self.output_dir / str(study + "_reverse.mat")
        # Command to convert the transformation matrices
        convert_command = [
            "convert_xfm",
            "-omat",
            str(reverse_transformation_matrix),
            "-inverse",
            str(transformation_matrix),
        ]

        try:
            self.v("Converting transformation matrices...", convert_command)
            subprocess.run(convert_command, check=True)
            self.v("Transformation matrices converted successfully.")
        except subprocess.CalledProcessError as e:
            self.v("Error converting transformation matrices:", e)
        # Inverse registration using FLIRT with the saved transformation matrix
        cmd = [
            "flirt",
            "-in",
            str(input_image),
            "-ref",
            str(ref_image),
            "-out",
            str(output_image),
            "-init",
            str(
                reverse_transformation_matrix
            ),  # Use the saved transformation matrix for inverse registration
            "-cost",
            "normmi",
            "-dof",
            "12",  # 6 degrees of freedom for rigid registration
            "-interp",
            "nearestneighbour",  # Interpolation method (adjust as needed)
            "-applyxfm",
        ]
        self.v("inverse transformation flirt...", cmd)
        subprocess.run(cmd, check=True)
        return output_image

    def reverse_reorientation(self, input_image: Path, ref_image: Path):
        assert input_image.exists(), f"{input_image} not found"
        assert ref_image.exists(), f"{ref_image} not found"
        # Reorient back to original space
        output_image = self.output_dir / str(
            Path(input_image.stem).stem + "_reoriented.nii.gz"
        )
        reorient_command = [
            "fslreorient2std",
            str(input_image),
            str(output_image),
            "-r",
            "-warp",
            str(ref_image),
        ]
        subprocess.run(reorient_command, check=True)

    def move_file(self, src_dir: Path, dest_dir: Path, file_name: str):
        # Construct the source and destination paths
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_path = src_dir / file_name
        dest_path = dest_dir / file_name

        try:
            # Use shutil.move to move the file
            shutil.move(src_path, dest_path)
            self.v(f"Moved {file_name} from {src_dir} to {dest_dir}")
        except FileNotFoundError as e:
            self.v(f"Error: {e}")

    def copy_file(self, src_dir: Path, dest_dir: Path, file_name: str):
        # Construct the source and destination paths
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_path = src_dir / file_name
        dest_path = dest_dir / file_name

        try:
            # Use shutil.copy to move the file
            shutil.copy(src_path, dest_path)
            self.v(f"Copy {file_name} from {src_dir} to {dest_dir}")
        except FileNotFoundError as e:
            self.v(f"Error: {e}")

    def reverse_preprocess(self, segmentation: Path = None):
        if segmentation is None and self.segmentation is not None:
            segmentation = self.segmentation
        assert segmentation.exists(), f"{segmentation} not found"

        self.reverse_1 = [
            self.inverse_registration_with_transformation(
                self.segmentation, tmat[-1], contrast[0], suid
            )
            for skull_stripped, tmat, contrast, suid in zip(
                self.skull_stripped, self.atlas, self.registered, self.studyuids
            )
        ]

        self.reverse_2 = [
            self.inverse_registration_with_transformation(
                image, tmat[-1], contrast, suid, ".nii.gz"
            )
            for image, tmat, contrast, suid in zip(
                self.reverse_1, self.registered, self.mr_contrasts, self.studyuids
            )
        ]

    @IO.Instance()
    @IO.Input('in_t1_data', 'nifti:mod=mr:type=t1', the='MR T1 image')
    @IO.Input('in_t1ce_data', 'nifti:mod=mr:type=t1ce', the='MR T1ce image')
    @IO.Input('in_t2_data', 'nifti:mod=mr:type=t2', the='MR T2 image')
    @IO.Input('in_flair_data', 'nifti:mod=mr:type=flair', the='MR FLAIR image')
    @IO.Output('out_t1_data', 't1_seg.nii.gz', 'nifti:mod=seg:type=t1:roi=NECROSIS,EDEMA,BRAIN,ENHANCING',
               the='t1 seg image')
    @IO.Output('out_t1ce_data', 't1ce_seg.nii.gz', 'nifti:mod=seg:type=t1ce:roi=NECROSIS,EDEMA,BRAIN,ENHANCING',
               the='t1ce seg image')
    @IO.Output('out_t2_data', 't2_seg.nii.gz', 'nifti:mod=seg:type=t2:roi=NECROSIS,EDEMA,BRAIN,ENHANCING',
               the='t2 seg image')
    @IO.Output('out_flair_data', 'flair_seg.nii.gz', 'nifti:mod=seg:type=flair:roi=NECROSIS,EDEMA,BRAIN,ENHANCING',
               the='FLAIR seg image')
    def task(self, instance: Instance, in_t1_data: InstanceData, in_t1ce_data: InstanceData,
             in_t2_data: InstanceData, in_flair_data: InstanceData, out_t1_data: InstanceData,
             out_t1ce_data: InstanceData, out_t2_data: InstanceData, out_flair_data: InstanceData):
        self.v("running task....")
        t1 = in_t1_data.abspath
        t1c = in_t1ce_data.abspath
        t2 = in_t2_data.abspath
        flair = in_flair_data.abspath

        output_dir = tempfile.mkdtemp()
        self._setup(t1, t1c, t2, flair, output_dir)
        self.v("running forward preprocessing....")
        os.environ['nnUNet_results'] = os.environ['WEIGHTS_FOLDER']
        self.forward_preprocess(predict=True)
        self.v("running reverse preprocessing....")
        self.reverse_preprocess()
        self.v(self.reverse_2)
        if len(self.reverse_2) > 0:
            shutil.copyfile(self.reverse_2[0], out_t1_data.abspath)
            shutil.copyfile(self.reverse_2[1], out_t1ce_data.abspath)
            shutil.copyfile(self.reverse_2[2], out_t2_data.abspath)
            shutil.copyfile(self.reverse_2[3], out_flair_data.abspath)
