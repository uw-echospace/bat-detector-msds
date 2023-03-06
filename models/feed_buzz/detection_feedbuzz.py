import os
import argparse

import pandas as pd

from models.detection_interface import DetectionInterface
import models.feed_buzz.feed_buzz_helper as fbh

class FeedBuzz(DetectionInterface):
    def __init__(self, peak_distance, peak_th, template_dict, num_matches_threshold, buzz_feed_range, alpha, COMPARE_TP):
        self.peak_distance = peak_distance
        self.peak_th = peak_th
        self.template_dict = template_dict
        self.num_matches_threshold = num_matches_threshold
        self.buzz_feed_range = buzz_feed_range
        self.alpha = alpha
        self.COMPARE_TP = COMPARE_TP #question: how to incorporate a path here?

    def run(self, audio_file) -> pd.DataFrame: # TODO: type annotations
        rois_df = fbh.run_multiple_template_matching(
                                            audio_file,
                                            self.peak_distance,
                                            self.peak_th,
                                            self.template_dict,
                                            self.num_matches_threshold, 
                                            self.buzz_feed_range, 
                                            self.alpha,
                                            self.COMPARE_TP)
        
        return rois_df
        

    def get_name(self): # TODO: do we really need this? config passes in the name
        return "model_interface"