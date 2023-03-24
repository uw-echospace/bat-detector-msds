import os

import numpy as np
from pathlib import Path

import librosa
import soundfile


def generate_segments(audio_file: Path, output_dir: Path, start_time: float, duration: float):
    """
    Segments audio_file into clips of duration length and saves them to output_dir.
    start_time: seconds
    duration: seconds
    """

    ip_audio, sampling_rate = librosa.load(audio_file, sr=None)

    # Convert to sampled units
    ip_start = int(start_time * sampling_rate)
    ip_duration = int(duration * sampling_rate)
    ip_end = len(ip_audio)

    output_files = []

    # for the length of the duration, process the audio into duration length clips
    for sub_start in range(ip_start, ip_end, ip_duration):
        sub_end = np.minimum(sub_start + ip_duration, ip_end)

        sub_length = sub_end - sub_start
        op_audio = np.zeros(int(sub_length), dtype=ip_audio.dtype)
        op_audio[:sub_length] = ip_audio[sub_start:sub_end]

        # For file names, convert back to seconds 
        op_file = os.path.basename(audio_file.name).replace(" ", "_")
        start_seconds = start_time / sampling_rate
        end_seconds = sub_start / sampling_rate
        op_file_en = "__{:.2f}".format(start_seconds) + "_" + "{:.2f}".format(end_seconds)
        op_file = op_file[:-4] + op_file_en + ".wav"
        
        op_path = os.path.join(output_dir, op_file)
        output_files.append({
            "audio_file": op_path, 
            "offset": start_time + (sub_start/sampling_rate),
        })
                
        # TODO: ensure 16 bitdepth is correct
        # TODO: maybe make this configurable parameter?
        soundfile.write(op_path, op_audio, sampling_rate, subtype='PCM_16') 

    return output_files 