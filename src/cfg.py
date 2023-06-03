from pathlib import Path
import os

from models.bat_call_detector.model_detector import BatCallDetector

# Contains configurable paramters for analysis portion of the pipeline. Anything that is not a CLI 
# argument (stuff that doesn't interact with the host OS) should be in here.

def get_config():
    """
    detection_threshold: float, ranges from 0.0 to 1.0.
        Cut-off probability for BatDetector2

    spec_slices: boolean.
        Used for visualisation

    chunk_size: int. 
        If files greater than this amount (seconds) they will be broken down into small chunks

    model_path: Path.
        Path where the pretrained model resides

    time_expansion_factor: int.
        The time expansion factor used for all files (default is 1)

    quiet: boolean.
        Minimize output printing

    cnn_features: boolean.
        Extracts CNN call features

    peak_distance: float.
        Required minimal temporal distance (>= 0) in seconds between neighbouring
        peaks. If set to `None`, the minimum temporal resolution will be used.
        The minimal temporal resolution is given by the array tn and depends on the parameters
        used to compute the spectrogram.

    peak_threshold: float, ranges -1 to 1.
        Threshold applied to find peaks in the cross-correlation array

    template_dict_path: Path.
        Location where the template pickle file is stored.

    num_matches_threshold: int, ranges 0 to the total number of templates.
        The number of template that matches the detected area of interest(aoi). The smaller this number is, the
        fewer templates that the detected aoi has to match in order to be returned as a confirmed feeding buzz.

    buzz_feed_range: float, in milisecond. ranges 0.0 to 1.0.
        The distance between two consecutive feeding buzz.

    alpha: int, ranges from 0 to 1.
        A tunable parameter to find the surrounding feeding buzzes identified by similar templates.

    """
    return {
        # TODO: document
        "time_expansion_factor": 1.0,
        
        # Offset (seconds) from the beginning of the audio file to start processing
        "start_time": 0.0,

        # Input audio is divided into segments of this duration (seconds), each processed individually
        "segment_duration": 30.0, 
        
        "models": [ 
            BatCallDetector( 
                # TODO: all parameters 
                detection_threshold=0.5,
                spec_slices=False,
                chunk_size=2, 
                model_path=f"{os.path.dirname(__file__)}/models/bat_call_detector/batdetect2/models/Net2DFast_UK_same.pth.tar",
                time_expansion_factor=1.0,
                quiet=False,
                cnn_features=True,
                peak_distance=0.05,
                peak_threshold=0.25,
                template_dict_path=f"{os.path.dirname(__file__)}/models/bat_call_detector/templates/template_dict.pickle",
                num_matches_threshold=2,
                buzz_feed_range=0.15,
                alpha=1,
            ),
        #    AnotherDetector(
        #         param1="param1",
        #         param2="param2",  
        #     )
        ]
    } 