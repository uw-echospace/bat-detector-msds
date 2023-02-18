# for connecting to azure storage account
from azure.storage.blob import BlobClient 
from azure.storage.blob import ContainerClient

# for showing progress bar
from tqdm import tqdm

# plotting spectrogram
import matplotlib.pyplot as plt
from scipy.io import wavfile

#from scipy import signal
from matplotlib.pyplot import figure
import torchaudio
import torch

# others
import os 
from pathlib import Path


def data_download(connectString:str, containerName:str, destPath:Path):
    """
    Download wav and txt file from Azure blob storage to local machine.
    """
    myContainer = ContainerClient.from_connection_string(conn_str=connectString, container_name=containerName)
    blob_list = myContainer.list_blobs()

    for blob in blob_list:
        name = blob.name
        print(name)
        # combining blob with local path,  type(filename) == pathlib.PosixPath
        filename = (destPath/name) 
        
        # creating parent directory if does not exist
        filename.parent.mkdir(parents=True,exist_ok=True )

        with filename.open("wb") as my_local_blob:
            try:
                # show download progress bar
                size = blob.size
                with tqdm.wrapattr(my_local_blob, "write", total=size) as file_obj:
                    print(" Writing file,{}, to local".format(name))
                    download_stream = myContainer.get_blob_client(blob).download_blob()
                    download_stream.readinto(file_obj)
            except:
                print("Writing to file failed.")
    return 

def convert_spectrogram(wavPath:Path, destPath:Path):
    """
    Convert individual .wav files into spectrogram and save as .png in destDir
    """
    samplingFrequency, signalData = wavfile.read(wavPath)

    # TODO: include melfilter bank here for smoothing
    spec = torchaudio.compliance.kaldi.fbank(signalData,
                                            htk_compat=True,
                                            sample_frequency=samplingFrequency,
                                            use_energy=False,
                                            window_type='hanning',
                                            num_mel_bins=128,
                                            dither=0.0,
                                            high_freq=12000,
                                            low_freq=500)
    spec = torch.flipud(spec.T).numpy()

    fig = plt.figure(figsize = (2,2))
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    Pxx, freqs, bins, im = plt.specgram(spec,Fs=samplingFrequency,NFFT=300,noverlap=100 )
    fig.savefig(f"{destPath}.png")
    plt.close()

def save_png(wavDir:Path, destDir:Path):
    """
    Go through the list of .wav file in the directory, convert into spectrogram and save as .png
    """
    wavList = wavDir.glob(f"*.wav")
    wavInfos = [{'path': t, 'prefix': t.stem[:3], 'wavname': t.stem[3:]} for t in list(wavList)]
    
    for wav in wavInfos:
        destName = destDir/wav['wavname']
        convert_spectrogram(wav['path'], destName)

if __name__ == "__main__":
    # global variables 
    globconnectString = """DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=kirstngcapstone;AccountKey=6AL8uFsyPJWwSZbXRChqEdVW55JkYBnWENVyuGiizw0V7Iv83x8g5FxSKD0Mb/KWbLFtQQcofwae+AStgw9yew==;
    BlobEndpoint=https://kirstngcapstone.blob.core.windows.net/;FileEndpoint=https://kirstngcapstone.file.core.windows.net/;
    QueueEndpoint=https://kirstngcapstone.queue.core.windows.net/;TableEndpoint=https://kirstngcapstone.table.core.windows.net/"""

    globcontainerName = "annotated-data"

    # testing variables for save_png
    wavDir = Path("/Users/kirsteenng/Desktop/UW/DATA 590/individual_spec/2021_09_09")
    destDir = Path("/Users/kirsteenng/Desktop/UW/DATA 590/individual_png/2021_09_09")
    

    mydestPath = Path('/Users/kirsteenng/Desktop/UW/DATA 590/test_download/')
    #data_download(connectString=globconnectString, containerName=globcontainerName, destPath=mydestPath)
    #convert_spectrogram("/Users/kirsteenng/Desktop/UW/DATA 590/individual spectrogram/hf_20221012_030000.WAV28.wav")
    save_png(wavDir,destDir)