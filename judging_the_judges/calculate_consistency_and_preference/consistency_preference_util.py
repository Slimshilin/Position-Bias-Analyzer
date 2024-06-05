from subeval.smp import load,dump,double_log
import pandas as pd
import numpy as np
import os
import csv
import re
from collections import defaultdict

def aggregate_judge_data(directory_path, nopt):
    """
    Load TSV files, aggregate judge evaluations, extract answers from evaluations, 
    and append the extracted answers to the DataFrame.

    Run this function after obtaining the results of running LLM-as-a-Judge.
    DO NOT revise the resulting file names (but feel free to move the whole resulting folder around)
    
    Parameters:
    - directory_path: str, the path to the directory containing the result files.
    - nopt: the Option mode (i.e., the number of options) when running LLM-as-a-Judge. Now the code supports 2 or 3.
    
    Returns:
    - pd.DataFrame, aggregated data with additional columns for extracted answers.
    """
    dfs = []  # To store dataframes for each file
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(f"_{nopt}.tsv"):
                # Extract the judge's name from the file name
                judge_name = file.split('_')[1] # Assuming 'record_[judge]_[nopt].tsv' format
                
                # Load the TSV file
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path, delimiter='\t')
                
                # Rename to faciliate extraction
                judge_col_name = f"judge_{judge_name}"
                df.rename(columns={df.columns[-1]: judge_col_name}, inplace=True)
                
                dfs.append(df)
    
    if dfs:
        # Identify common columns excluding new judge-rep columns
        common_cols = list(set.intersection(*(set(df.columns) for df in dfs)))
        common_cols = [col for col in common_cols if 'judge_' not in col]
        
        # Initialize aggregated DataFrame with the first DataFrame
        aggregated_data = dfs[0]
        
        # Iteratively merge remaining DataFrames
        for df in dfs[1:]:
            aggregated_data = pd.merge(aggregated_data, df, on=common_cols, how='outer')

        # Just in case: Drop rows where 'A' and 'B' are equal -- same model, will not generate any response
        aggregated_data = aggregated_data[aggregated_data['A'] != aggregated_data['B']]
        
        return aggregated_data
    else:
        raise ValueError("No dfs extracted")


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
    Applies the match_answer function to extract answers from all judgment columns
    and creates new columns named as 'extracted__{judge}'.

    Run this function after running `aggregate_judge_data` which gives you the judgment columns that
    start with "judge_".
    
    Parameters:
    - data: pd.DataFrame, the DataFrame containing the judge's full judgments (including choice an reasoning).
    
    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the extracted answers.
    """
    # Iterate through columns to find those with judge decisions across repetitions
    for col in data.columns:
        if 'judge_' in col:
            # Construct the new column name for the extracted answer
            new_col_name = f"extracted_{col[len('judge_'):]}"
            # Apply the match_answer function to each row in the column
            data[new_col_name] = [match_answer(ans, benchmark) for ans in data[col]]
    
    # Fill NA with 'N' - this faciliates extraction successful rate (or error) calculation
    extracted_columns = [col for col in data.columns if col.startswith('extracted')]
    data[extracted_columns] = data[extracted_columns].fillna('N')
    return data
