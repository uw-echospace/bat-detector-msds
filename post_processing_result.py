import pandas as pd
from pathlib import Path


#TODO1: identify all .json files in directory.
input_dir = Path("/Users/kirsteenng/Desktop/UW/DATA 590/workflow/3_annotation")
json_list = input_dir.glob(f"*.json")

full_list = pd.DataFrame()

for indv in json_list:
    curr = pd.json_normalize(pd.read_json(indv)['annotation'])

    curr_start_time = float(indv.stem[22:][:-4])
    curr['start_time'] =  curr['start_time'] + curr_start_time
    curr['end_time'] =  curr['end_time'] + curr_start_time
    
    full_list = pd.concat([full_list,curr],ignore_index = True)

full_list.sort_values(by = ['start_time'],inplace = True, ascending = True)
full_list.to_csv('/Users/kirsteenng/Desktop/UW/DATA 590/workflow/full_list.csv')
