from pathlib import Path

from models.bat_detect2.detection_other import BatDetect2 

config = {
    # TODO: decide what should be CLI args to the application and what should remain in configuration
    "audio_file_path": Path("data/audio/2020-08-01_21-00-00.wav"), 
    "output_path": Path("output"),
    "time_expansion_factor": 1.0,
    "start_time": 0.0,
    "segment_duration": 30.0,
    "models": [
        {
            "name": "batdetect2",
            "model": BatDetect2( # TODO: comment on what all these parameters are
                detection_threshold=0.5,
                spec_slices=False,
                chunk_size=2, 
                model_path="batdetect2/models/Net2DFast_UK_same.pth.tar",
                time_expansion_factor=1.0, # TODO: what did Kirsteen use?
                quiet=False,
                cnn_features=True,
                # TODO: might need to update model path to be relative to repo root
            ),
        }
    ]
} 