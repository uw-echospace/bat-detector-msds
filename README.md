
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
git submodule update --init --recursive
```
```
pip install -r requirements.txt
```

## Usage
The following invokation generates a TSV in `output_dir` containing all detections:
```
python src/cli.py audio.wav output_dir/
```

Additionally, you can specify the number of processes used to process the audio and generate a CSV instead of TSV:
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

The pipeline executes the `run()` method of every model present in that aforementioned `models` list in `src/cfg.py`.


## Update feedbuzz detection templates
To identify feedbuzzes, this repository uses a technique called [template matching](https://en.wikipedia.org/wiki/Template_matching). We offer an initial set of templates, that is stored in `src/models/bat_call_detector/templates/template_dict.pickle`  that could perform decently for feeding buzz from bat calls native to Seattle, Washington. The templates are generated based on the following steps:

1. An individual feeding buzz is identified in an audio recording. The time and frequency of the feeding buzz are being identified manually.
2. Run `generate_template()` function in `src/models/bat_call_detector/feed_buzz_helper.py` to generate template based on the time and frequency identified above.
3. The template will be saved in a pickle object.

User can see what are the templates stored in the template_dict.pikle by running `load_template()` function in `src/models/bat_call_detector/feed_buzz_helper.py`. However, the details below are the templates used in the current pipeline.

| Template      | Audio File Name | Time (s) | Frequency  (kHz) |
| ----------- | ----------- |----------- |----------- |
| 1   | 20210910_030000_time2303_LFbuzz.wav | (9.762, 10.059) | (14532.7, 29760.3)
| 2   | 20210910_033000.wav  | (70.637, 71.328) |(19745, 28638.2)
| 3   | 20210910_033000.wav  |(620.663, 620.854) |(12434.9,29910.9)
| 4   | 20210910_033000.wav  |(898.079, 898.368) |(11426.6, 25205.9)
| 5   | 20210910_030000.wav  |(608.139, 608.452) |(14328.0,30138.3)
| 6   | 20210910_030000.wav  |(744.961, 745.0877) |(10375.5, 47430.83)
| 7   | 20210910_030000.wav  |(1065.034, 1065.228) |(14328, 25691.7)
| 8   | 20211016_030000.wav  |(1611.886, 1612.014) |(19214.9,53801.6)
| 9   | 20211016_030000.wav  |(1717.383, 1717.518) |(19762.8, 46442.7)
| 10  | 20211016_030000.wav |(1728.248, 1728.397 )|(20751, 52865.6)


User can choose to update these templates as a way to improve the performance of feeding buzz detection. Follow the steps below to update the templates. Note that all the functions mentioned below are in `src/models/bat_call_detector/feed_buzz_helper.py`
1. Run `load_template()`  to assess existing template. 
2. Run `remove_template()`  to remove any unwanted template.
3. Run `generate_template()`  to generate new templates and save it to existing or new template dictionary.

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
Dr. Wu-Jung Lee -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Aditya Krishna -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Juan Sebastian Ulloa -- Author of [scikit-maad](https://github.com/macaodha/batdetect2) \
Oisin Mac Aodha -- Author of [Bat Detect 2](https://github.com/macaodha/batdetect2)
