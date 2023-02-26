import numpy as np
from pathlib import Path

# TODO: figure out weird dependency on bat_detect.audio_utils

# TODO: write to output directory
# TODO: annotate types
def segment_audio(audio_file: Path, output_dir: Path, st_time, duration, time_expansion_factor):
    sampling_rate, ip_audio = au.load_audio_file_preprocess(
        audio_file, time_expansion_factor
    )

    # check the total duration of the input audio file
    audio_length = ip_audio.shape[0]/sampling_rate

    end_time = len(ip_audio) * args["time_expansion_factor"] # cut the whole audio file 
    
    audio_segments = []
    # for the length of the duration, process the audio into duration length clips
    for sub_sample_index in np.arange(st_time,end_time,duration):
        en_time = sub_sample_index + duration
        st_samp = int(sub_sample_index * sampling_rate)
        en_samp = np.minimum(int(en_time * sampling_rate), ip_audio.shape[0])

        try: 
            op_audio = np.zeros(int(sampling_rate * duration, dtype=ip_audio.dtype))
            op_audio[: en_samp - st_samp] = ip_audio[st_samp:en_samp]

            audio_segments.append({
                "begin_time": st_time,
                "sampling_rate": sampling_rate,
                "audio": op_audio
            })
        except:
            print("There is an error.")
            break

    return audio_segments