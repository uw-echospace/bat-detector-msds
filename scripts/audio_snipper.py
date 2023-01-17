#!/usr/bin/env python

from pathlib import Path

import pandas as pd
import type_enforced
from scipy.io import wavfile


@type_enforced.Enforcer

def generateAllSnippets(srcDir: Path, tsvDir: Path, destDir: Path, windowSize: int, tsvExt: str):
    # Get all TSVs in destDir
    tsvPaths = tsvDir.glob(f"*.{tsvExt}")
    tsvInfos = [{'path': t, 'prefix': t.stem[:3], 'wavname': t.stem[3:]} for t in tsvPaths]

    for tsvInfo in tsvInfos:
        # Get the corresponding wav
        wavPath = srcDir / tsvInfo['wavname']
        generateSnippetsForTSV(wavPath, destDir, tsvInfo, windowSize)


def generateSnippetsForTSV(wavPath: Path, destDir: Path, tsvInfo: dict, windowSize: int):
    df = pd.read_csv(tsvInfo['path'], sep='\t', header=0)
    df = df[df['View'] == 'Waveform 1']

    for _, row in df.iterrows():
        # TSV HEADER: Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Occupancy
        start = row['Begin Time (s)']
        end = row['End Time (s)']

        sample_rate, wav = wavfile.read(wavPath)
        snipped_wav = singleSnippet(sample_rate, wav, start, end, windowSize)

        # Write the snipped data to destination directory
        wavfile.write(
                filename=destDir / f"{tsvInfo['prefix']}{row['Selection']}.wav", 
                rate=sample_rate, 
                data=snipped_wav) 


def singleSnippet(sample_rate, wav, start, end, windowSize):
    start = int(start * sample_rate)
    end = int(end * sample_rate)

    if windowSize is not None: 
        halfWindowSize = int(windowSize * sample_rate) // 2
        midpoint = start + end // 2
        start = midpoint - halfWindowSize
        end = midpoint + halfWindowSize
    
    return wav[start:end]


