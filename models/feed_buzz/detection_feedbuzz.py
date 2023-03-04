import os
import argparse

import pandas as pd

from models.detection_interface import DetectionInterface

class FeedBuzz(DetectionInterface):
    def __init__(self):
        pass

    def run(self, sampleing_rate, audio) -> pd.DataFrame: # TODO: type annotations
        pass

    def get_name(self): # TODO: do we really need this? config passes in the name
        return "model_interface"