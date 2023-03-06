import os

import numpy as np
from pathlib import Path

import librosa
import soundfile


def generate_segments(audio_file: Path, output_dir: Path, start_time: float, duration: float):

    ip_audio, sampling_rate = librosa.load(audio_file, sr=None)

    #audio_length = ip_audio.shape[0]/sampling_rate

    # TODO: is it right to excplude time_expansion_factor here? We leave it to the modles to decide 
    #       how to preprocess their own audio
    end_time = len(ip_audio) # * args["time_expansion_factor"] # cut the whole audio file 
    
    output_files = []
    # for the length of the duration, process the audio into duration length clips
    for sub_sample_index in np.arange(start_time, end_time, duration):
        en_time = sub_sample_index + duration
        st_samp = int(sub_sample_index * sampling_rate)
        en_samp = np.minimum(int(en_time * sampling_rate), ip_audio.shape[0])

        try: 
            op_audio = np.zeros(int(sampling_rate * duration), dtype=ip_audio.dtype)
            op_audio[: en_samp - st_samp] = ip_audio[st_samp:en_samp]


            op_file = os.path.basename(audio_file.name).replace(" ", "_")
            op_file_en = "__{:.2f}".format(start_time) + "_" + "{:.2f}".format(sub_sample_index)
            op_file = op_file[:-4] + op_file_en + ".wav"
            
            op_path = os.path.join(output_dir, op_file)
            output_files.append({"audio_file": op_path, "offset": start_time + sub_sample_index})
            
            soundfile.write(op_path, op_audio, sampling_rate, subtype='PCM_16') # TODO: ensure 16 bitdepth is correct

            #print("\n{}\tIP: ".format(ii) + os.path.basename(ip_path))
            #print("\tOP: " + os.path.basename(op_path))
        except:
            # TODO: investigate why this occurs once during segmentation of an audio 
            print("There is an error.")
            break

    return output_files 