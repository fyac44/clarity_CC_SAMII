""" Demo of how to run and evaluate the baseline

Use this script to understand how to run and use the tools provided.
    
(This script serves as a replacement for the top-level bash scripts
that were used in the initial release and which are now obsolete.)
"""
import sys

import argparse
import pandas as pd

import matlab.engine

import calculate_SI
import calculate_SAMII
import check_data
import run_HL_processing
import predict_intel
import report_rmse_score


DATAROOT = "../data/clarity_data/"  # Root directory where clarity data is stored


def main(skip_checks=False, run_baseline=False, run_samii=False, open_set=False, n_signals=None):
    """Run and evaluate the MSBG/MBSTOI baseline and BEZ2018/SAMII system."""

    """
    1. Check the integrity of the data, if requested

    This step runs through the installed CPC signal data and makes sure that all files
    are present. If any are missing the script will stop immediately. We recommend
    using this check so that the script does not fail after several hours of computation.add()

    2. Use track 1 (close set) or track 2 (open set)
    """

    if not skip_checks and check_data.main(DATAROOT):
        print("Files missing.")
        sys.exit()

    if open_set:
        dataset = 'train_indep'
    else:
        dataset = 'train'

    """
    2. Run the full baseline model, if requested

    The baseline model has two computational stages.
        i/ Signals are passed through the MSBG hearing loss model
        ii/ Hearing-loss simulated signals are evaluated with the MBSTOI intelligibility metric

    SAMII has two computational stages.
        i/ Signals are passed through the BEZ2018 peripheral model and information is calculated
        ii/ SAMII is calculated from information

    Be aware, these stages are quite slow and will run at about 2 or 3 signals per minute.
    """
    mbstoi_file = f"mbstoi.{dataset}.csv"
    samii_file = f"samii.{dataset}.csv"

    if run_baseline:
        print("Running hearing loss model processing")
        run_HL_processing.main(
            f"{DATAROOT}/metadata/scenes.CPC1_train.json",
            f"{DATAROOT}/metadata/listeners.CPC1_train.json",
            f"{DATAROOT}/metadata/CPC1.{dataset}.json",
            f"{DATAROOT}/clarity_data/scenes",
            f"{DATAROOT}/clarity_data/HA_outputs/{dataset}",
            f"{DATAROOT}/clarity_data/HA_outputs/{dataset}",
            n_signals,
        )

        print("Running MBSTOI intelligibility model")
        calculate_SI.main(
            f"{DATAROOT}/metadata/CPC1.{dataset}.json",
            f"{DATAROOT}/clarity_data/scenes",
            f"{DATAROOT}/clarity_data/HA_outputs/{dataset}",
            f"{mbstoi_file}",
            n_signals,
        )
    
    if run_samii:
        print("Running BEZ2018/MI")
        eng = matlab.engine.start_matlab()
        eng.CPC1_BEZ2018_SAMII("..", dataset, str(n_signals), nargout=0)
        eng.quit()

        print("Running SAMII")
        calculate_SAMII.main(
            f"{DATAROOT}/clarity_data/mutualinfo/{dataset}/",
            f"{DATAROOT}/metadata/CPC1.{dataset}.json",
            f"{samii_file}",
            n_signals,
        )

    """
    3. Check the MBSTOI and SAMII csv files. 
    
    If the baseline has not been run then it will run the evaluation 
    steps using the precomputed values that are available in example_data/

    If the proposed algorithm has not been run then it will deactivate the 
    mapping for samii
    """

    try:
        pd.read_csv(mbstoi_file)
    except:
        print("Warning: MBSTOI csv file not found. Using pre-computed values from example_data/")
        mbstoi_file = "example_data/mbstoi.train.baseline.csv"
    
    try:
        pd.read_csv(samii_file)
    except:
        print("Warning: SAMII csv file not found. No fitting for SAMII will be performed")
        run_samii = False 

    """
    4. Generate intelligibility predictions.
    
    The raw MBSTOI and SAMII values are mapped onto sentence intelligiblities (i.e., predictions of the 
    percentage of words that will be recognised correctly from 0 to 100)
    This mapping is performed with a sigmoid function whose parameters are learnt by
    performing a best fit from predicted MBSTOI or SAMII values to the actual intelligibilities. 
    Look at this stage to understand how to split the data into a training and development
    test set in a fair way.
    """

    print("Generate intelligibility predictions from raw MBSTOI values")
    predict_intel.main(
        "mbstoi",
        mbstoi_file,
        f"{DATAROOT}/metadata/CPC1.{dataset}.json",
        f"predictions_{dataset}_mbstoi.csv",
    )

    if run_samii:
        print("Generate intelligibility predictions from raw SAMII values")
        predict_intel.main(
            "samii",
            samii_file,
            f"{DATAROOT}/metadata/CPC1.{dataset}.json",
            f"predictions_{dataset}_samii.csv",
            )

    """
    5. Compute the RMS error score.

    The true intelligibilities are provided in the CPC1.train.json file and the 
    predicted intelligilities from the previous step are in predictions.csv
    This very simple step computes and report the overall RMS error score.
    This is the score that will be used to rank submissions.
    """
    print("Score the predictions for MBSTOI")
    report_rmse_score.main(f"{DATAROOT}/metadata/CPC1.{dataset}.json", f"predictions_{dataset}_mbstoi.csv")

    print("Score the predictions for SAMII")
    report_rmse_score.main(f"{DATAROOT}/metadata/CPC1.{dataset}.json", f"predictions_{dataset}_samii.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip the data integrity checks",
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Run full baseline MSBG+MBSTOI model (slow)",
    )
    parser.add_argument(
        "--run_samii",
        action="store_true",
        help="Run full BEZ2018+SAMII model (slow)",
    )
    parser.add_argument(
        "--open_set",
        action="store_true",
        help="Run the open set of the data",
    )
    parser.add_argument(
        "--n_signals",
        type=int,
        default=None,
        help="Number of signals to process. If not set then processes all.",
    )
    args = parser.parse_args()
    main(args.skip_checks, args.run_baseline, args.run_samii, args.open_set, args.n_signals)
