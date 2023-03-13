from pathlib import Path

from models.bat_call_detector.model_detector import BatCallDetector

# Contains configurable paramters for analysis portion of the pipeline. Anything that is not a CLI 
# argument (stuff that doesn't interact with the host OS) should be in here.
def get_config():
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
                model_path="src/models/bat_call_detector/batdetect2/models/Net2DFast_UK_same.pth.tar",
                time_expansion_factor=1.0,
                quiet=False,
                cnn_features=True,
                peak_distance=0.05,
                peak_threshold=0.25,
                template_dict_path="src/models/bat_call_detector/templates/template_dict.pickle",
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