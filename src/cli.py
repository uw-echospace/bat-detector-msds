import argparse
import os
from pathlib import Path

import numpy as np

# set ython path
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "models/bat_call_detector/batdetect2/"))

from pipeline import pipeline as pipeline

# TODO: make MODELS config?
from src.cfg import get_config  

# TODO: add models to CLI, but for now, just use all of the by default

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_audio",
        type=str,
        help="Input directory containing the audio files",
    )

    return vars(parser.parse_args())

# if main
if __name__ == "__main__":
    args = parse_args()

    #print("Input directory   : " + args["input_directory"])
    #print("Output directory  : " + args["output_directory"])
    #print("Start time        : {}".format(args["start_time"]))
    #print("Output duration   : {}".format(args["output_duration"]))
    #print("Audio files found : {}".format(len(ip_files)))

    cfg = get_config()
    cfg["audio_file_path"] = Path(args["input_audio"])

    pipeline.run(cfg) # TODO: return value?
    