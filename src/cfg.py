from pathlib import Path

from models.bat_call_detector.model_detector import BatCallDetector

# TODO: consider converting this to JSON or YAML file and putting in root of repo

# Contrains configurable paramters for ML portion of the pipeline. Anything that is not a CLI 
# argument (stuff that doesn't interact with the host OS) should be in here.
def get_config():
    return {
        # TODO: comment on what all these parameters are
        "time_expansion_factor": 1.0,
        "start_time": 0.0,
        "segment_duration": 30.0, #units in seconds
        "models": [ 
            # TODO: can we just make this a list of objects instead of list of dicts?
            {
                "model": BatCallDetector( 
                    detection_threshold=0.5,
                    spec_slices=False,
                    chunk_size=2, 
                    model_path="models/bat_call_detector/batdetect2/models/Net2DFast_UK_same.pth.tar",
                    time_expansion_factor=1.0, # TODO: what did Kirsteen use?
                    quiet=False,
                    cnn_features=True,
                    peak_distance=0.05,
                    peak_threshold=0.25,
                    template_dict_path="models/bat_call_detector/templates/template_dict.pickle",
                    num_matches_threshold=2,
                    buzz_feed_range=0.15,
                    alpha=1,
                    # TODO: might need to update model path to be relative to repo root
                ),
            },
            # {
            #     "model": AnotherDetector(
            #         param1="param1",
            #         param2="param2",  
            #     )
            # }
        ]
    } 