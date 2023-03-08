from pathlib import Path

#from models.bat_detect2.model_batdetect2 import BatDetect2 

from models.bat_call_detector.model_detector import BatCallDetector

def get_config():
    return {
        # TODO: decide what should be CLI args to the application and what should remain in configuration
        # TODO: gets input file from CLI right now...
        #"audio_file_path": Path("data/audio/2020-08-01_21-00-00.wav"), 
        "csv_output_path": Path("output"),
        "tmp_output_path": Path("output/tmp"),
        "time_expansion_factor": 1.0,
        "start_time": 0.0,
        "segment_duration": 30.0,
        "models": [
            {
                "name": "batdetect2",
                "model": BatCallDetector( # TODO: comment on what all these parameters are
                    detection_threshold=0.5,
                    spec_slices=False,
                    chunk_size=2, 
                    model_path="models/bat_call_detector/batdetect2/models/Net2DFast_UK_same.pth.tar",
                    time_expansion_factor=1.0, # TODO: what did Kirsteen use?
                    quiet=False,
                    cnn_features=True,
                    peak_distance=0.05,
                    peak_threshold=0.25,
                    template_dict_path="/Users/kirsteenng/Desktop/UW/DATA 590/bat-detector-msds/pipeline/template_dict.pickle", #TODO: is this the right place to put?
                    num_matches_threshold=2,
                    buzz_feed_range=0.15,
                    alpha=1,
                    # TODO: might need to update model path to be relative to repo root
                ),
            },
            # {
            #     "name": "feed-buzz-detector",
            #     "model": #FeedBuzzDetector() # TODO 
            # }
        ]
    } 