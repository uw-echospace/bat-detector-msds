#!/usr/bin/env python

from pathlib import Path
import argparse
import pandas as pd

from scipy.io import wavfile

# Gather our code in a main() function
def main(args):
    print(args)

    # TODO call generateAllSnippets
    # TODO convert all times to milliseconds
    generateAllSnippets(
            args['SourceAudioDir'], 
            args['SourceTSVDir'], 
            args['DestinationDir'], 
            args['window'], 
            args['tsv_extension'])

def generateAllSnippets(srcDir: Path, tsvDir: Path, destDir: Path, windowSize: int, tsvExt: str):
    # Get all TSVs in destDir
    tsvPaths = tsvDir.glob(f"*.{tsvExt}")
    tsvInfos = [{'path': t, 'prefix': t.stem[:3], 'wavname': t.stem[3:]} for t in tsvPaths]

    for tsvInfo in tsvInfos:
        # Get the corresponding wav
        wavPath = srcDir / tsvInfo['wavname'] + ".wav"
        generateSnippetsForTSV(wavPath, destDir, tsvInfo, windowSize)


def generateSnippetsForTSV(wavPath: Path, destDir: Path, tsvInfo: dict, windowSize: int):
    df = pd.read_csv(tsvInfo['path'], sep='\t', header=0)
    df = df[df['View'] == 'Waveform 1']

    for _, row in df.iterrows():
        # TSV HEADER: Selection	View	Channel	Begin Time (s)	End Time (s)	Low Freq (Hz)	High Freq (Hz)	Occupancy
        start = row['Start Time (s)']
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser( 
        description = "Segment audio file into snippets that contain bat calls. The audio files are assumed to be in .wav format. The TSV files are assumed to be in the format output by the RavenPro.",
    )

    # TODO Specify your real parameters here.
    parser.add_argument("SourceAudioDir", help = "The directory containing the audio files to segment.")
    parser.add_argument("SourceTSVDir", help = "The directory containing the TSVs associated with each audio file.")
    parser.add_argument("DestinationDir", help = "The directory to output the audio segments")

    # TODO: default arguments
    parser.add_argument(
        "-w",
        "--window",
        help="The size of the window in milliseconds. If not present, use the exact intervals defined in the TSV file.",
        action="store_true")

    parser.add_argument(
        "-t",
        "--tsv_extension",
        help="The expected file extension of the TSV files. Defaults to 'txt'.",
        default="txt",
        action="store_true")
    
    args = parser.parse_args()
    
    main(args)
        