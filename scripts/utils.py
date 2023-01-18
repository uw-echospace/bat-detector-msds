# required libraries: azure storage, tqdm

# for connecting to azure storage account
from azure.storage.blob import BlobClient 
from azure.storage.blob import ContainerClient

# for showing progress bar
from tqdm import tqdm

# others
import os 
from pathlib import Path


def data_download(connectString:str, containerName:str, destPath:Path):
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

if __name__ == "__main__":
    # global variables 
    globconnectString = """DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=kirstngcapstone;AccountKey=6AL8uFsyPJWwSZbXRChqEdVW55JkYBnWENVyuGiizw0V7Iv83x8g5FxSKD0Mb/KWbLFtQQcofwae+AStgw9yew==;
    BlobEndpoint=https://kirstngcapstone.blob.core.windows.net/;FileEndpoint=https://kirstngcapstone.file.core.windows.net/;
    QueueEndpoint=https://kirstngcapstone.queue.core.windows.net/;TableEndpoint=https://kirstngcapstone.table.core.windows.net/"""

    globcontainerName = "annotated-data"

    mydestPath = Path('/Users/kirsteenng/Desktop/UW/DATA 590/test_download/')
    data_download(connectString=globconnectString, containerName=globcontainerName, destPath=mydestPath)