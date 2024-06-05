'''
Helper functions for calcualting repetitional consistency.
'''

import pandas as pd
import os
import re

def load_repetition_data(directory_path, nopt):
    """
    Load TSV files from specified directory path and stack the judge's decisions
    across repetitions into new columns.
    
    Parameters:
    - directory_path: str, the path to the directory containing the result folders.
    
    Returns:
    - pd.DataFrame, the aggregated data with new columns for each judge's decision
      across different repetitions in a form of {judge}-rep{repetition number}.
      
    Assumption:
    - Under the directory_path, sub-directories are named "Repetition{repetition number}_xxxxxxxx"
    - Under the sub-directories, each include a tsv named "record_{judge}_{nopt}.tsv", {nopt} for number of options (Option Mode, i.e., 2 or 3)
    - All repetitions across all judges lie in the same structure of columns (except for the last column with {judge} name)
    - Only the last column, {judge}, is varied among different judges, but is the same for all repetitions
    """
    # Initialize an empty DataFrame for the aggregated data.
    aggregated_data = None
    # Data structure to hold each dataframe before merging
    dfs = []
    
    # Iterate over directories and files within the specified path.
    for root, dirs, files in os.walk(directory_path):
        # Extract the repetition number from the directory name.
        parts = root.split(os.sep)
        if parts[-1].startswith('Repetition'):
            rep_number = parts[-1].replace('Repetition', '')[0]  # Assuming 'RepetitionX_' format
            
            for file in files:
                if file.endswith(f"_{str(nopt)}.tsv"):
                    # Extract the judge name from the file name.
                    judge_name = file.split('_')[1]  # Assuming 'record_[judge]_[nopt].tsv' format
                    new_col_name = f"{judge_name}-rep{rep_number}"
                    
                    # Load the TSV file.
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, delimiter='\t')
                    # Rename the judge's decision column to include the judge name and repetition number.
                    df.rename(columns={df.columns[-1]: new_col_name}, inplace=True)
                    dfs.append(df)

    # Merge all DataFrames on common columns, if any dfs were added
    if dfs:
        # Identify common columns excluding new judge-rep columns
        common_cols = list(set.intersection(*(set(df.columns) for df in dfs)))
        common_cols = [col for col in common_cols if '-rep' not in col]
        
        # Initialize aggregated DataFrame with the first DataFrame
        aggregated_data = dfs[0]
        
        # Iteratively merge remaining DataFrames
        for df in dfs[1:]:
            aggregated_data = pd.merge(aggregated_data, df, on=common_cols, how='outer')
    
    return aggregated_data


def match_answer(s, benchmark):
    '''
    Extracts the judgment choice (A/B/C/D) from the given string based on a specific patterns.

    This should align with the prompt settings. This should also be exactly the same function as in `analyze_util.py` in the `subeval` module.

    Feel free to revise the regular expressions according to your needs.
    '''
    if benchmark == "MTBench":
        if result := re.findall(r'\[\[([ABCD])\]\]', s): 
            return result[0]
        else:
            return None
    elif benchmark == "DevBench":
        if result := re.findall('(?:选择：|Choice: )([ABCD])', s): 
            return result[0]
        else:
            return None
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")

def extract_answer(data, benchmark):
    """
    Applies the match_answer function to extract answers from all repetition columns
    and creates new columns named as 'extracted-{judge}-rep{repetition}'.
    
    Parameters:
    - data: pd.DataFrame, the DataFrame containing the judge's decisions across repetitions.
    
    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the extracted answers.
    """
    # Iterate through columns to find those with judge decisions across repetitions
    for col in data.columns:
        if '-rep' in col:
            # Construct the new column name for the extracted answer
            new_col_name = f"extracted-{col}"
            # Apply the match_answer function to each row in the column
            data[new_col_name] = [match_answer(ans, benchmark) for ans in data[col]]
    
    return data