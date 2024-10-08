#!/bin/zsh

# Ask the user for the file name and search term
recoverFolder=$1
sdUnit=$2
runModel=$3
genFig=$4
shouldCSV=$5
numProcesses=$6

echo "Input Directory: $recoverFolder"/"$sdUnit";
echo "Output CSV name: bd2__"$recoverFolder"_"$sdUnit""
echo "Output Directory: output_dir/"$recoverFolder"/"$sdUnit""

. /home/adkris/miniconda3/etc/profile.d/conda.sh
conda activate bat_msds

python3 src/batdt2_pipeline.py "$recoverFolder" "$sdUnit" output_dir/"$recoverFolder" output/tmp $runModel $genFig $shouldCSV $numProcesses
