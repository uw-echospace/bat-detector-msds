#!/bin/zsh

# Ask the user for the file name and search term
recoverPath=$1
sdUnit=$2

recoverFolder=$(basename "$recoverPath")

echo "Input Directory: $recoverPath"/"$sdUnit";
echo "Output Directory: output_dir/"$recoverFolder"/"$sdUnit""

. /home/adkris/miniconda3/etc/profile.d/conda.sh
conda activate bat_msds

python3 src/batdt2_pipeline.py "$recoverPath"/"$sdUnit" bd2__"$recoverFolder"_"$sdUnit" output_dir/"$recoverFolder" output/tmp --run_model --csv --night_only --individual_files