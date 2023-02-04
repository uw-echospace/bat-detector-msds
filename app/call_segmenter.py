from pathlib import Path


# CSC: don't know if we need
def segment_bat_calls(wav_path: Path, dest_dir: Path):
    """
    1. input big wave file 
    2. apply band energy detection
    3. output text file (in RavenPro format) containing list of bat call locations within the large wav 

    CSC: I outputting a text file instead of a bunch of small wavs is perhaps a premature optimization, but it IS an optimization. 
         This method reduces disk IO and I suspect may speed things up quite a lot. 
    """
    pass