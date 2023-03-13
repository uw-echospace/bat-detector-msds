
# MSDS Capstone Project: Bats!!!

## Overview
This repository offers the following features:
* Detection of bat search, social, and feedbuzz calls
* A fast, customizable pipeline for automating application of the aforementioned detectors

# Usage
## Setup
We recommend using Python 3.9.x. Other versions may work, but they're not tested. We also recommend creating a Python [virtual environment](https://docs.python.org/3/library/venv.html).

From repository root, run the following commands:
```
git submodule --init --recursive
```
```
pip install -r requirements.txt
```

## Usage
The following invokation generates a TSV in `output_dir` containing all detections:
```
python src/cli.py audio.wav output_dir/
```

Additionally, you can specific the number of processes used to process the audio and generate a CSV instead of TSV:
```
python src/cli.py --csv --num_processes=4 audio.wav output_dir/
```

## Analytics Configuration
All of the analytical parameters are accessible in `src/cfg.py`. Have a look! 
## Adding custom detectors
`src/cfg.py` is also where new, custom detectors can be added. To add your own detector to the pipeline:
1. Create a new class in `src/models/` that inherits from `src/models/detection_interface.py`
2. Override `DetectionInterface`'s `run()` and `get_name()` methods
3. Add your model's constructer to `src/cfg.py` in the `models` list, passing in any parameters needed in the constructor. 

The pipeline executres the `run()` method of every model present in that aforementioned `models` list in `src/cfg.py`.


## Update feedbuzz detection templates
To identify feedbuzzes, this repository uses a technique called [tempalte matching](https://en.wikipedia.org/wiki/Template_matching). We offer an initial set of templates that perform decently for feedbuzz detection in Seattle, WA. To update these templates:
1. TODO: 1
2. TODO: 2

## Help
```
python src/cli.py --help
```

# Analysis
TODO
## Detectors and Results
TODO


# Acknowledgements
Dr. Wu-Jung Lee -- Univeristy of Washington [EchoSpace](https://uw-echospace.github.io) \
Aditya Krishna -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Maad person TODO \
Oisin Mac Aodha -- [Bat Detect 2](https://github.com/macaodha/batdetect2) \
