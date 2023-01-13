# required libraries: azure storage, tqdm

# for connecting to azure storage account
from azure.storage.blob import BlobClient  # must ensure this libray in requirement.txt
from azure.storage.blob import ContainerClient

# for showing progress bar
from tqdm import tqdm

# others
import os 

# global variables 
globconnectString = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=kirstngcapstone;AccountKey=6AL8uFsyPJWwSZbXRChqEdVW55JkYBnWENVyuGiizw0V7Iv83x8g5FxSKD0Mb/KWbLFtQQcofwae+AStgw9yew==;BlobEndpoint=https://kirstngcapstone.blob.core.windows.net/;FileEndpoint=https://kirstngcapstone.file.core.windows.net/;QueueEndpoint=https://kirstngcapstone.queue.core.windows.net/;TableEndpoint=https://kirstngcapstone.table.core.windows.net/"

globcontainerName = "annotated-data"

#globdestPath = "/Users/kirsteenng/Desktop/UW/DATA 590/test_download/"



def data_download(connectString = globconnectString, containerName = globcontainerName , destPath):
    myContainer = ContainerClient.from_connection_string(conn_str= connectString, container_name= containerName)
    blob_list = myContainer.list_blobs()
    suffix = '.WAV'
    wav_list = []
    for blob in blob_list:
        if blob.name.endswith(suffix):
            print(blob.name + '\n')
            wav_list.append(blob.name) # storing list of .wav files to be iterated
    
    
    for wav in wav_list:

        # create connection to individual blob
        targetWav = BlobClient.from_connection_string(conn_str= connectString, container_name=containerName, blob_name = wav)

        # create directory on local machine if not exist for respective .wav file
        filename = os.path.join(destPath, wav)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as my_local_blob:
            try:
                size = targetWav.get_blob_properties()['size']

                # show download progress bar
                with tqdm.wrapattr(my_local_blob, "write", total=size) as file_obj:
                    download_stream = targetWav.download_blob()
                    print("Download complete. Writing file,{}, to local".format(wav))
                    download_stream.readinto(file_obj)
            except:
                print("Download failed.")

    return 


mydestPath = 'DECLARE YOUR PATH'
data_download(destPath = mydestPath)