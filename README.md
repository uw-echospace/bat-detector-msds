
# MSDS Capstone Project: Bats!!!

## Overview
This repository offers the following features:
* Detection of bat search, social, and feedbuzz calls
* A fast, customizable pipeline for automating application of the aforementioned detectors

# Usage
## Setup
We recommend using Python 3.9.x. Other versions may work, but they're not tested. We also recommend using conda [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/).

From repository root, run the following commands:
```
git submodule update --init --recursive
```
```
pip install -r requirements.txt
```
**Recommended**
```
conda env create -f environment.yml
conda activate bat_msds 
```

## Directory structure
- `field_records`: This is where I store the field records saved in `uw-echospace/ubna-field` so that I can assign recordings to location based on recover-DATE, Audiomoth name, and SD card #. The location is also an argument I can provide manually in cases where the field records are not updated for the detector to find the location of the most recent recordings.
- `ims`: unchanged images folder from the 2023 MSDS capstone team's work.
- `notebooks`: This is where I (try to) store well-documented notebooks to show team members how to run the pipeline or how to assemble detections into activity grids.
- `output_dir`: This is where I store detection `.csv` files or activity grid plots
  - `UBNA_202309` contains last year's preliminary array detection .csv files from our work with YeonJoon.
  - `cumulative_plots` contains large activity grids spanning several months divided across locations and frequency groups.
    - `cumulative_activity__*Bat_30T.png` pertains to 1 week of detections collected from region close to bat boxes.
    - `cumulative_activity__*Carp_30T.png` pertains to beginning of 2024 season till most recent data from Carp Pond (since Carp Pond was newly added)
    - All other `cumulative_activity__*[LOCATION]_30T.png` files pertain to all data recorded from recoveries in the year 2024.
    - `[HF|LF][LOCATION]` tag implies k-means clustering was implemented on normalized Welch PSD curves of each detected bat call to separate detections into the HF/LF group.
  - `mic_array_test_[DATE]` contains any detection files separated by file name from recording tests of the array microphone and the Audiomoth.
  - `recover-DATE` contains sub-folders either `[SD_CARD]` or `[LOCATION_NAME]` corresponding to the detections and activity grid for a specific period of time recorded from an Audiomoth that was deployed in the corresponding location (found either by SD card or manual input).
- `scripts` contains `.sh` files for automating the detection process. These are not currently being used since calling the `python src/batdt2_pipeline.py ...` was simple enough.
- `src` contains the model scripts along with the 2023 MSDS team's pipeline code.
  - `batdt2_pipeline.py` is where I have added all the code for invoking the pipeline, running the detector, generating activity grids.
  - `file_dealer.py` is what I use to look through all files and then identify which files are good for detection and which files to skip.


## Usage for deployment-based data
1) `python3 src/file_dealer.py "/mnt/ubna_data_04" "output_dir" "ubna_data_04_collected_audio_records.csv"`
   - Argument 1 `mnt/ubna_data_04` is the external drive that is read.
   - Argument 2 `output_dir` is where the final output is saved.
   - Argument 3 `ubna_data_04_collected_audio_records.csv ` is the name of the output file
   - This reads all recordings (recursively within sub-directories) and uses the pattern `recover-DATE/SD_NUM/*.WAV` to extract DATE of recovery, SD_CARD that the files were saved into, and the filepaths. Then `exiftool` and `field_records` are used to understand what location the files come from, which Audiomoth (A/B/C...) recorded the file, what metadata does the file have (sampling-rate/battery-voltage/recording-duration/...) which also tells us if the file is a good for feeding into detection. All of these details are stored into a final output.csv with the specified name.
   - This line is entered AFTER all data for a recover-DATE has been uploaded. The resulting output.csv will have information for all the latest recordings which will be used to generate detections for each location.

2) `python3 src/batdt2_pipeline.py --recover_folder="recover-20240927" --sd_unit="008" --site='Telephone Field' --recording_start='00:00' --recording_end='23:59' --cycle_length=600 --duration=300 --output_directory='output_dir/recover-20240927' --run_model --generate_fig --csv`
   - `recover_folder` and `sd_unit` is needed to find the files to process.
   - `site` is used to assign the location. This is optional and if not provided, the code will use the field_records to find the corresponding location. If the field records do not have this information, then the `sd_unit` argument will be used and "(Site not found in field records)" will be the location name in plots.
   - `recording_start` and `recording_end` is used as part of a `between_time()` operation to select the time span per night that data is processed. To include all 24hrs, we have set it as '0:00' and '23:59' (24:00 is not valid). For a separate project, I used '0:00' and '16:00' and only that time span per night was selected to be processed which sped up the pipeline.
   - `cycle_length` and `duration` are used to place constraints on the file duration of "good" files. These are also used to scale the number of detections. If 10 calls were detected in 5 minutes of a 10-min cycle. Then this is scaled to 20 calls in 10-min.
   - `output_directory` is where a `[SD_NUM|LOCATION]` folder is created and the output `detections.csv`, `activity__*.csv` and `activity__*.png` files are saved under the created folder.
   - `--run_model` runs the detections to create the output `detections.csv` file.
   - `--generate_figs` creates the intermediate and final data formats to visualize activity.
   - `--csv` makes the output detections file a `.csv`. Or else, it would be a RavenPro readable `.txt` file.
   - Other arguments exist and have been explained in the code but are not used in the current pipeline.
   - This pipeline currently takes 18 hours from the addition in k-means clustering adding 1s per file segment. There are typically 11000 file segments for 1 week of data where each segment is 30-secs long.

3) `python3 src/batdt2_pipeline.py ... --sd_unit="007" --site='Carp Pond' ...`
   - The format is kept the same and `site` and `sd_unit` are changed.
   - The command is repeated until all locations have a detection file and activity grid generated.
   - Once this is done, we wait until next week's data is recovered and uploaded and start from command 1.
  
## Usage for location-year-month data 
1) `python3 src/batdt2_pipeline.py --site='Telephone Field' --year='2022' --month=='August' --recording_start='00:00' --recording_end='23:59' --output_directory='output_dir' --run_model --csv`
   - Generates detection.csv files for each file recorded from the year 2022 in August at Telephone Field regardless of SD card, Audiomoth name or other factors.
   - `recording_start` and `recording_end` are used the same way as before
   - `duration` is default-set at 1795 when argument unspecifed. 1795 is the file duration we used for majority of the 2022-recording season.
   - `output_directory` will be `output_dir/"Telephone Field"`. This is where each detections.csv file will be saved.
   - Each detections.csv file will be named in the format of `bd2__[SITE-TAG]_[FILENAME].csv`
  
This usage assumed that there exists a `ubna_data_*_collected_audio_records.csv` which is the output of `python3 src/file_dealer.py "/mnt/ubna_data_*" "output_dir" "ubna_data_*_collected_audio_records.csv"`. This file should be in `output_dir` and hold information about the recordings you wish to process.

## Usage for single-file data 
1) `python3 src/batdt2_pipeline.py --input_audio='/mnt/ubna_data_04/recover-20240927/UBNA_008' --output_directory='output_dir' --run_model --csv`
   - `input_audio` can be directory or single-file (a directory is provided above). The detector will run on each file and save `batdetect2_pipeline_[FILENAME].csv` in the provided `output_directory`
   - No other arguments need to be provided. Intended for direct-use.


# Previous versions
- See the [msds-2023](https://github.com/uw-echospace/bat-detector-msds/tree/msds-2023) branch for the first version built by students in the 2023 Masters in Data Science (MSDS) program.


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
Corbin Charpentier -- University of Washington Masters in Data Science Program \
Kirsteen Ng -- University of Washington Masters in Data Science Program \
Ernesto Cediel -- University of Washington Masters in Data Science Program \
Dr. Wu-Jung Lee -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Aditya Krishna -- University of Washington [EchoSpace](https://uw-echospace.github.io) \
Juan Sebastian Ulloa -- Author of [scikit-maad](https://github.com/macaodha/batdetect2) \
Oisin Mac Aodha -- Author of [Bat Detect 2](https://github.com/macaodha/batdetect2)
