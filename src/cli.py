import argparse
import os
from pathlib import Path

import numpy as np

# set python path to correctly use batdeteect2 submodule
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src/models/bat_call_detector/batdetect2/"))

from pipeline import pipeline as pipeline

from src.cfg import get_config  

def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_audio",
        type=str,
        help="the WAV file to process",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="the directory to write the output to",
        default="output",
    )
    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="the directory to write the temporary files to",
        default="output/tmp",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generate CSV instead of TSV",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    cfg = get_config()
    cfg["should_csv"] = args["csv"]
    cfg["output_dir"] = Path(args["output_directory"])
    cfg["tmp_dir"] = Path(args["tmp_directory"])
    cfg["audio_file"] = Path(args["input_audio"])
    cfg["num_processes"] = args["num_processes"]

    _ = pipeline.run(cfg)
    