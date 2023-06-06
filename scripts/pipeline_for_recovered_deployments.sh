#!/bin/zsh

# Ask the user for the file name and search term
recoverPath=$1
sdUnit=$2

recoverFolder=$(basename "$recoverPath")

echo "Input Directory: $recoverPath"/"$sdUnit";
echo "Output CSV name: batdetect2_pipeline__"$recoverFolder"_"$sdUnit".csv"
echo "Output Directory: output_dir/"$recoverFolder"/"$sdUnit""

. /home/adkris/miniconda3/etc/profile.d/conda.sh
conda activate bat_msds

python src/batdt2_pipeline.py "$recoverPath"/"$sdUnit" batdetect2_pipeline__"$recoverFolder"_"$sdUnit".csv output_dir/"$recoverFolder"/"$sdUnit" output/tmp

# python src/batdt2_pipeline.py $recover_folder_path/$sd_unit2 batdetect2_pipeline__$recover_folder_$sd_unit2.csv output_dir/$recover_folder/$sd_unit2 output/tmp

# python src/batdt2_pipeline.py $recover_folder_path/$sd_unit3 batdetect2_pipeline__$recover_folder_$sd_unit3.csv output_dir/$recover_folder/$sd_unit3 output/tmp

