
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

# Deeper dive into the models inside the library
We have created a software combining **BatDetect2** and **scikit-maad** to increase the accuracy and efficiency in bat calls and feeding buzz detection. The pipeline will then be programmed to run in parallel processes to increase efficiency.
## BatDetect 2 
**BatDetect2** is a convolutional neural network based open-source pipeline for detecting ultrasonic, full-spectrum, search-phase calls produced by echolocating bats. The model first converts a raw audio file into a spectrogram and uses a sliding window method to identify the pieces of spectrogram that contains bat calls. 

![BatDetect2_example](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/BatDetect2_example.png?raw=true)

Example output of BatDetect2.

## Scikit-maad (Spectrogram Template Matching)
**Scikit-maad** is a Python package that specializes in quantitative analysis of environmental audio recording. Given that feeding buzzes and ordinary bat calls have different shapes in the spectrogram and leveraging the stereotypical shape of feeding buzzes, we use multiple feed buzz templates and a template matching function provided in the package, proving to be effective in identifying feeding buzzes amongst bat calls.

![BatCall_example](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/bat_call_example.png?raw=true)

(a) A group of bat calls have consistent frequency between each call

![FeedingBuzz_example](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/feeding_buzz_example.png?raw=true)

(b) A feeding buzz is identified as a sudden dip in calls.

![TemplateMatching_example](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/template_matching_example.png?raw=true)

Example output of template matching from scikit-maad using only one template. The bounding boxes in top image show the feeding buzz identified. The correlation coefficient chart below indicates the coefficient of this file with the template used. Note the three peaks in the chart corresponds to the bounding boxes in the top chart. 

Our model combines the results of multiple templates (10 templates) that are passed to each spectrogram. Given that this resluts in many different potential feeding buzz detections, we use a voting system among all of these detections to choose the final feeding buzz identifications. Currently our voting threshold is: 2.


## Pipeline Workflow 

The following diagram describes the overall pipeline of our model:

![PipelineWorkflow](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/workflow.jpg?raw=true)

# Analysis
## Model Evaluation
We evaluate our model based on calculating Recall and Precision metrics using one audio wav file: 20210910_030000.wav that contains more than 3000 bat calls.
 
### Bat calls
One tunable parameter in this bat call model is the probability threshold, which refers to the detection probability computed by the model. The higher the probability, the more confident the model is in identifying the target as a bat call. We found that the Recall-Precision for bat calls is most optimized around threshold=0.44, with both recall and precision rate around 0.85. 

![PRCurve_BatCalls](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/PRCurve_BatCalls.png?raw=true)

### Feeding buzzes
We created a method of combining threshold tuning and filtering false positives using the result from the bat call pipeline to improve our recall and precision rate from 0.25 to 0.6 using two templates (number of templates=2). The threshold that provides the most balanced outcome is 0.26. This threshold represents the correlation coefficient between the target and template. 

![PRCurve_FeedingBuzzes](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/PRCurve_FeedingBuzz.png?raw=true)

## Results
Based on the table below, our pipeline has increased the Precision by 73%, Recall by 140% for bat call detection and the Computation time by 10% for a 30-minute audio wav file.

![ResultsTable](https://github.com/uw-echospace/bat-detector-msds/blob/main/ims/ResultsTable.png?raw=true)

*The value for precision is not available for feeding buzzes because there is no labelled data in the manual process

Computation times gains are calculated on the specific improvement that our sponsor will observe, so it has to be taken with care. We explain why:
1. Our sponsor currently uses RavenPro. For batch processing on Mac the software limits batch processing to no more than approximately 16 files for 16GB RAM and 8 files for 8GB files, hence, they were forced to use a slower Linux machine to be able to batch process the amount of files they require. This machine is what the currently use and it's the baseline we use of 2 minutes 36 seconds per file.

2. Our library can be run on any OS, specifically in the faster Mac machine they have available, we know that for a similar Mac Book Pro M1X with 64GB RAM it takes 2 minutes 12 seconds to run. This will be the processing time they will observe per file. 


# Acknowledgements
Dr. Wu-Jung Lee -- Univeristy of Washington [EchoSpace](https://uw-echospace.github.io) \
Aditya Krishna -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Juan Sebastian Ulloa -- Author of [scikit-maad](https://github.com/macaodha/batdetect2) \
Oisin Mac Aodha -- Author of [Bat Detect 2](https://github.com/macaodha/batdetect2)
