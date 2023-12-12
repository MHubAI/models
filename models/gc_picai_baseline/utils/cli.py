"""
--------------------------------------------------
Mhub / DIAG - CLI for the PICAI baseline Algorithm
--------------------------------------------------

--------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
--------------------------------------------------
"""

import argparse
from pathlib import Path
from process import csPCaAlgorithm as PicaiClassifier


def run_classifier(t2: Path, adc: Path, hbv: Path, cancer_likelihood_json: Path, cancer_detection_heatmap: Path):
    # Initialize classifier object
    classifier = PicaiClassifier()

    # Specify input files (the order is important!)
    classifier.scan_paths = [
        t2,
        adc,
        hbv,
    ]

    # Specify output files
    classifier.cspca_detection_map_path = cancer_detection_heatmap
    classifier.case_confidence_path = cancer_likelihood_json

    # Run the classifier on the input images
    classifier.process()


def run_classifier_cli():
    parser = argparse.ArgumentParser("CLI to run the PICAI baseline classifier")
    parser.add_argument("input_t2", type=str, help="input T2 weighted prostate MR image (MHA)")
    parser.add_argument("input_adc", type=str, help="input ADC prostate MR image (MHA")
    parser.add_argument("input_hbv", type=str, help="input HBV prostate MR image (MHA)")
    parser.add_argument("output_cancer_likelihood_json", type=str, help="output JSON file with PICAI baseline prostate cancer likelihood (JSON)")
    parser.add_argument("output_cancer_detection_heatmap", type=str, help="output heatmap indicating prostate cancer likelihood (MHA)")
    args = parser.parse_args()
    run_classifier(
        t2=Path(args.input_t2),
        adc=Path(args.input_adc),
        hbv=Path(args.input_hbv),
        cancer_likelihood_json=Path(args.output_cancer_likelihood_json),
        cancer_detection_heatmap=Path(args.output_cancer_detection_heatmap),
    )


if __name__ == "__main__":
    run_classifier_cli()
