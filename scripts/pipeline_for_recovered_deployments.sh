#!/bin/zsh

# Ask the user for the file name and search term
read -p "Enter recover folder name: " recover_folder_path

recover_folder = $(basename -- recover_folder_path)

conda activate bat_msds

for sd_unit in $recover_folder_path/*/ ; do
    python src/batdt2_pipeline.py $recover_folder_path/$sd_unit batdetect2_pipeline__$recover_folder_$sd_unit.csv output_dir/$recover_folder/$sd_unit output/tmp
done