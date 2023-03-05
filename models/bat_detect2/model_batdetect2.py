import os
import argparse
import pandas as pd

from models.detection_interface import DetectionInterface

import bat_detect.utils.detector_utils as du

class BatDetect2(DetectionInterface):

    # TODO: what constructor params are needed?
    def __init__(self, detection_threshold, spec_slices, chunk_size, model_path, time_expansion_factor, quiet, cnn_features):
        self.detection_threshold = detection_threshold
        self.spec_slices = spec_slices
        self.chunk_size = chunk_size
        self.model_path = model_path
        self.time_expansion_factor = time_expansion_factor
        self.quiet = quiet
        self.cnn_features = cnn_features

    def get_name(self):
        return "batdetect2"

    def run(self, audio_file):
        print('Loading model: ' + self.model_path)
        model, params = du.load_model(self.model_path)

        model_output = du.process_file(
            audio_file=audio_file, 
            model=model, 
            params=params, 
            args= {
                'detection_threshold': self.detection_threshold,
                'spec_slices': self.spec_slices,
                'chunk_size': self.chunk_size,
                'quiet': self.quiet,
                'spec_features' : False,
                'cnn_features': self.cnn_features,
            },
            time_exp=self.time_expansion_factor,
        )

        # TODO: what else needs to go in here?
        out_df = pd.DataFrame({
            "start_time": [],
            "end_time": [],
            "low_freq": [],
            "high_freq": [],
        })

        # TODO: move to base class
        annotations = model_output['pred_dict']['annotation']
        for annotation in annotations:
            out_df = out_df.append({
                "start_time": annotation['start_time'],
                "end_time": annotation['end_time'],
                "low_freq": annotation['low_freq'],
                "high_freq": annotation['high_freq'],
            }, ignore_index=True)

        return out_df 
