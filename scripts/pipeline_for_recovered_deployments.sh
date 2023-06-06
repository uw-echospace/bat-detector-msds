#!/bin/zsh

# Ask the user for the file name and search term
recoverPath=$1
sdUnit=$2
runModel=$3
genFig=$4

recoverFolder=$(basename "$recoverPath")

echo "Input Directory: $recoverFolder"/"$sdUnit";
echo "Output CSV name: batdetect2_pipeline__"$recoverFolder"_"$sdUnit".csv"
echo "Output Directory: output_dir/"$recoverFolder"/"$sdUnit""

. /home/adkris/miniconda3/etc/profile.d/conda.sh
conda activate bat_msds

python src/batdt2_pipeline.py "$recoverPath"/"$sdUnit" batdetect2_pipeline__"$recoverFolder"_"$sdUnit".csv output_dir/"$recoverFolder"/"$sdUnit" output/tmp $runModel $genFig