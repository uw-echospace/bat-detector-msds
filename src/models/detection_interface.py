import pandas as pd
from pathlib import Path

class DetectionInterface:
    """
    To add a model to the pipeline, create a new class that inherits from this class.
    The new class should implement the run() method and the get_name() method.
    run() should return a pandas dataframe with columns defined in `utils/utils.py`.
    """
    def __init__(self):
        pass

    def run(self, audio_file:Path) -> pd.DataFrame: # TODO: type annotations
        """
        The pipeline calls this method to execute the model on the given audio file.
        Should return a pandas dataframe with columns defined in `utils/utils.py`.
        """
        pass

    def get_name(self):
        """
        Should return the name of the model inheriting from this class.
        The pipeline calls this method to get the name of the model.
        """
        return "TODO: implement get_name()"
