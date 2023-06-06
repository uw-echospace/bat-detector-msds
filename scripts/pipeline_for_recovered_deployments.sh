#!/bin/zsh

# Ask the user for the file name and search term
read -p "Enter recover folder path: " recover_folder_path
read -p "Enter SD card label: " sd_unit

recover_folder=$(basename "$recover_folder_path")

. /home/adkris/miniconda3/etc/profile.d/conda.sh
conda activate bat_msds

python src/batdt2_pipeline.py $recover_folder_path/$sd_unit batdetect2_pipeline__$recover_folder_$sd_unit.csv output_dir/$recover_folder/$sd_unit output/tmp

# python src/batdt2_pipeline.py $recover_folder_path/$sd_unit2 batdetect2_pipeline__$recover_folder_$sd_unit2.csv output_dir/$recover_folder/$sd_unit2 output/tmp

# python src/batdt2_pipeline.py $recover_folder_path/$sd_unit3 batdetect2_pipeline__$recover_folder_$sd_unit3.csv output_dir/$recover_folder/$sd_unit3 output/tmp

