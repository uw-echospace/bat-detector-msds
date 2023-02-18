"""
You can clip your files so that they are shorter using this script. 
You need to specify the locations of the input files and where you want the 
shorter files to be saved.

There are additional settings that allow you to specify the output duration 
and where in the file you start clipping from.
"""

import argparse
import os
from pathlib import Path

import numpy as np

# TODO: CSC: can we import from bat_detect directly?
import submodules.batdetect2.bat_detect.utils.audio_utils as au
import submodules.batdetect2.bat_detect.utils.wavfile as wavfile


def parse_args():
    info_str = (
        "\nScript that extracts smaller segment of audio from a larger file.\n"
        + " Place the files that should be clipped into the input directory.\n"
    )

    print(info_str)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_directory",
        type=str,
        help="Input directory containing the audio files",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Output directory the clipped audio files",
    )
    parser.add_argument(
        "--output_duration",
        default=2.0,
        type=float,
        help="Length of output clipped file (default is 2 seconds)",
    )
    parser.add_argument(
        "--start_time",
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

    audio_files = list(Path(args["input_directory"]).rglob("*.wav")) + list(
        Path(args["input_directory"]).rglob("*.WAV")
    )
    ip_files = [os.path.join(aa.parent, aa.name) for aa in audio_files]

    print("Input directory   : " + args["input_directory"])
    print("Output directory  : " + args["output_directory"])
    print("Start time        : {}".format(args["start_time"]))
    print("Output duration   : {}".format(args["output_duration"]))
    print("Audio files found : {}".format(len(ip_files)))

    if len(ip_files) == 0:
        return False

    if not os.path.isdir(os.path.dirname(args["output_directory"])):
        os.makedirs(os.path.dirname(args["output_directory"]))

    for ii, ip_path in enumerate(ip_files):
        sampling_rate, ip_audio = au.load_audio_file_preprocess(
            ip_path, args["time_expansion_factor"]
        )

        # check the total duration of the input audio file
        audio_length = ip_audio.shape[0]/sampling_rate
        print("File duration: {}".format(audio_length))

        st_time = args["start_time"]
        duration = args["output_duration"]
        end_time = len(ip_audio) * args["time_expansion_factor"] # cut the whole audio file 
       
        # for the length of the duration, process the audio into duration length clips
        for sub_sample_index in np.arange(st_time,end_time,duration):
            
            en_time = sub_sample_index + duration
            st_samp = int(sub_sample_index * sampling_rate)
            en_samp = np.minimum(int(en_time * sampling_rate), ip_audio.shape[0])

            print("sub_ending time: "+ str(en_samp))

            try: 
                op_audio = np.zeros(int(sampling_rate * args["output_duration"]), dtype=ip_audio.dtype)
                op_audio[: en_samp - st_samp] = ip_audio[st_samp:en_samp]


                op_file = os.path.basename(ip_path).replace(" ", "_")
                op_file_en = "__{:.2f}".format(st_time) + "_" + "{:.2f}".format(sub_sample_index)
                op_file = op_file[:-4] + op_file_en + ".wav"

                op_path = os.path.join(args["output_directory"], op_file)
                wavfile.write(op_path, sampling_rate, op_audio)

                print("\n{}\tIP: ".format(ii) + os.path.basename(ip_path))
                print("\tOP: " + os.path.basename(op_path))

            except:
                print("There is an error.")
                break


if __name__ == "__main__":
    main()
