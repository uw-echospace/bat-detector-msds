#!/usr/bin/env python

from pathlib import Path
import pandas as pd
from scipy.io import wavfile
import torchaudio
import numpy as np
import torch

def generateAllSnippets(wavDir: Path, tsvDir: Path, destDir: Path, windowSize: float = None, tsvExt: str = "txt"):
    """
    Takes path to a folder of wav files and a folder of RavenPro tsv files, and generates snippets of audio from the wav files
    """
    # Get all TSVs in destDir
    tsvPaths = tsvDir.glob(f"*.{tsvExt}")

    tsvInfos = [{'path': t, 'prefix': t.stem[:3], 'wavname': t.stem[3:]} for t in list(tsvPaths)]

    for tsvInfo in tsvInfos:
        # Get the corresponding wav
        wavPath = wavDir / tsvInfo['wavname']
        generateSnippetsForTSV(wavPath, destDir, tsvInfo, windowSize)


def generateSnippetsForTSV(wavPath: Path, destDir: Path, tsvInfo: dict, windowSize: float,expansion_ratio:int =10):
    """
    Generate all snippets for a single RavenPro tsv file
    """
    df = pd.read_csv(tsvInfo['path'], sep='\t', header=0)
    df = df[df['View'] == 'Waveform 1']

    for _, row in df.iterrows():
        # TSV HEADER from RavenPro: Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Occupancy
        start = row['Begin Time (s)']
        end = row['End Time (s)']

        #sample_rate, wav = wavfile.read(wavPath)

        # Consider the normalised and unnormalised option for reading wav file.
        wav,sample_rate = torchaudio.load(wavPath, normalize=True)
        target_sample_rate = sample_rate / expansion_ratio
    
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        resampled_wav = resample(wav)

        snipped_wav = _singleSnippet(sample_rate, resampled_wav, start, end, windowSize)
        
        # Write the snipped data to destination directory
        wavfile.write(
                filename=destDir / f"{tsvInfo['prefix']}{tsvInfo['wavname']}{row['Selection']}.wav", 
                rate=sample_rate, 
                data=snipped_wav) 


def _singleSnippet(sample_rate:int, wav:torch.Tensor, start:float, end:float, windowSize:float)->np.array:
    """
    Actually snip the audio
    """
    start = int(start * sample_rate)
    end = int(end * sample_rate)


    # TODO: there will beproblems if window crosses the beginning of end of the wav file.
    #       this is not likely to happen, however.
    if windowSize is not None: 
        halfWindowSize = int(windowSize * sample_rate) // 2
        midpoint = start + end // 2
        start = midpoint - halfWindowSize
        end = midpoint + halfWindowSize
    
    return np.array(wav[[0][0]][start:end])

# TODO: - write tests that pair snipper with blob downloader.
#       - remove this test code
if __name__ == "__main__":

    # To test, swap these paths with paths to your local data
    generateAllSnippets( 
        Path("/Users/kirsteenng/Desktop/UW/DATA 590/wav_annotation/2022_10_12_wav"),
        Path("/Users/kirsteenng/Desktop/UW/DATA 590/wav_annotation/2022_10_12_txt"),
        Path("/Users/kirsteenng/Desktop/UW/DATA 590/temp/2022_10_12_2"),
        0.1
    )
