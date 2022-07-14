import argparse
import csv
import json
import logging
import sys
import os

from clarity_core.config import CONFIG
from tqdm import tqdm

sys.path.append("../projects/SAMII")
from lib_samii import experiment


def main(signals_directory, metadata_filename, output_samii_file, nsignals=None):
    """Main entry point, being passed command line arguments.

    Args:
        signals_filename (str): name of json file containing signal_metadata
        output_sii_file (str): name of output samii csv file
        nsignals (int, optional): Process first N signals. Defaults to None, implying all.
    """
    
    data_list = os.listdir(signals_directory)
    pre_silence = CONFIG.pre_duration
    
    with open(metadata_filename) as json_metadata:
        metadata_raw = json.load(json_metadata)

    metadata = dict()
    for md in metadata_raw:
        md_key = md['signal']
        metadata[md_key] = md
        
    del metadata_raw

    f = open(output_samii_file, "a")
    writer = csv.writer(f)
    writer.writerow(["scene", "listener", "system", "SAMII"])

    # Process the first n signals if the nsignals parameter is set
    if nsignals and nsignals > 0:
        data_list = data_list[0:nsignals]

    plotit = 0

    for file in tqdm(data_list):
        filename = file.split('.')[0]
        with open(signals_directory+file) as json_data:
            data = json.load(json_data)
        
        listener = metadata[filename]['listener']
        system = metadata[filename]['system']
        scene = metadata[filename]['scene']

        logging.info(f"Running SI calculation: scene {scene}, listener {listener}")

        exp = experiment(data, filename, pre_silence)

        samii = exp.get_samii()

        if plotit < 1 and system == 'E010':
            exp.generate_plots('./projects/SAMII/figures/')
            plotit += 1

        writer.writerow([scene, listener, system, samii])
        f.flush()

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsignals", type=int, default=None)
    parser.add_argument("signals_directory", help="directory containing json files with signals' mutual information and entropy")
    parser.add_argument("metadata_filename", help="json file containing metadata of the signals")
    parser.add_argument("output_samii_file", help="name of output samii csv file")
    args = parser.parse_args()

    main(
        args.signals_directory,
        args.metadata_filename,
        args.output_samii_file,
        args.nsignals,
    )
