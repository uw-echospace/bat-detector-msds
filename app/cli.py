import argparse
import os
from pathlib import Path

import numpy as np

import bat_detect.utils.audio_utils as au
import bat_detect.utils.wavfile as wavfile

import pipeline

# TODO: add models to CLI, but for now, just use all of the by default

def parse_args():
    info_str = (
        "\nScript that extracts smaller segment of audio from a larger file.\n"
        + " Place the files that should be clipped into the input directory.\n"
    )

    print(info_str)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_audio",
        type=str,
        help="Input directory containing the audio files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./labels.txt",
        help="Output TSV containing all the labels"
    )
    parser.add_argument(
        "--segment_duration",
        default=2.0,
        type=float,
        help="Length of output clipped file (default is 2 seconds)",
    )
    parser.add_argument(
        "--segment_start_time",
        type=float,
        default=0.0,
        help="Start time from which the audio file is clipped (defult is 0.0)",
    ),
    parser.add_argument(
        "--time_expansion_factor",
        type=int,
        default=1,
        help="The time expansion factor used for all files (default is 1)",
    )
    return vars(parser.parse_args())

def main():
    args = parse_args()

    #print("Input directory   : " + args["input_directory"])
    #print("Output directory  : " + args["output_directory"])
    #print("Start time        : {}".format(args["start_time"]))
    #print("Output duration   : {}".format(args["output_duration"]))
    #print("Audio files found : {}".format(len(ip_files)))

    models = []

    labels = pipeline.run(
        models,
        Path(args["input_audio"]), 
        args["segment_duration"], 
        args["segment_start_time"], 
        args["time_expansion_factor"])

    
    # TODO: write labels to args["output_file"]

