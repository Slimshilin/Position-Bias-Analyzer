'''
These function calculates the repetitional consistency and standard deviation.

Note that the data preparation format is DIFFERENT than that for position bias.

You'll need to manually rename the resulting judgment output folders to let them be in such form: Repetition{repetition number}_xxxxxxxx
'''

from judging_the_judges.repetitional_consistency.repetitonal_consistency_util import load_repetition_data, extract_answer
import pandas as pd
import re
import os

def calculate_repetition_bias_for_judge(data, benchmark, save_to_directory=None):
    """
    Calculates the repetition bias for each judge and augments the DataFrame with repetitional consistency and standard deviations.

    Parameters:
    - data: pd.DataFrame, the DataFrame containing extracted answers for each judge across repetitions.
    - benchmark: MTBench or DevBench. Just a part of the file name.
    - save_to_directory: save to it if specified.

    Assumptions:
    - The data should have columns in a form of 'extracted-{judge}-rep{repetition}'
    - The swapped order of response is regarded as a different evaluation instance than the original one (i.e., one cmp_index per repetitional consistency calculation)
    - Since the number of cmp_index is same for all judges and all repetitions, the average of the consistency for each cmp_index gives the overall repetitional consistency for repetition bias.
    """

    # Pattern to extract the judge name from the column name
    pattern = re.compile(r'extracted-(.+?)-rep\d+')

    # Extracting all unique judge names from the columns
    judge_columns = [col for col in data.columns if col.startswith('extracted')]
    judges = set(pattern.match(col).group(1) for col in judge_columns if pattern.match(col))

    # Initialize a dictionary to hold the repetitional consistency and standard deviation for each judge
    judge_repetitional_consistency = {}

    for judge in judges:
        # Filter columns for the current judge using the extracted judge name
        judge_cols = [col for col in judge_columns if f'extracted-{judge}-rep' in col]

        # Calculate consistency for each cmp_index (i.e., for each row)
        data[f'repetitional_consistency_rate_{judge}'] = data.apply(lambda row: calculate_repetitional_consistency_rates(row[judge_cols]), axis=1)

        # Calculate standard deviation of repetitional consistencys for each cmp_index (i.e., for each row)
        data[f'repetitional_consistency_std_dev_{judge}'] = data.apply(lambda row: calculate_repetitional_consistency_std_dev(row[judge_cols]), axis=1)

        # Calculate repetitional consistency and standard deviation for the judge
        judge_repetitional_consistency[judge] = {
            'consistency_rate': data[f'repetitional_consistency_rate_{judge}'].mean(),
            'consistency_std_dev': data[f'repetitional_consistency_std_dev_{judge}'].mean()
        }

    # Convert the dictionary to a DataFrame
    result_df = pd.DataFrame.from_dict(judge_repetitional_consistency, orient='index')

    if save_to_directory:
        os.makedirs(save_to_directory, exist_ok=True)
        result_df.to_csv(os.path.join(save_to_directory, f'{benchmark}_repetition_consistency_results.csv'))

    print(f"{benchmark} repetitional consistency calculation complete.")

    return result_df

def calculate_repetitional_consistency_std_dev(judge_responses):
    """
    Helper function to calculate the standard deviation of the repetitional consistency of a judge's evaluation
    (recommended for row-level computation).
    
    Parameters:
    - judge_responses: pd.Series, the judge's responses across repetitions.
    
    Returns:
    - float: The standard deviation of the repetitional consistency.
    """
    mode_result = judge_responses.mode()
    if mode_result.empty:
        return None
    else:
        repetitional_consistency_rates = judge_responses.apply(lambda x: 1 if x == mode_result[0] else 0)
        return repetitional_consistency_rates.std()

def calculate_repetitional_consistency_rates(judge_responses):
    """
    Helper function to calculate the repetitional consistency of a judge's evalution instances
    (recommended for row-level computation).
    
    Parameters:
    - judge_responses: pd.Series, the judge's responses across repetitions.
    
    Returns:
    - float: The repetitional consistency.
    """
    # Count the frequency of each response
    response_counts = judge_responses.value_counts()
    # Calculate repetitional consistency as the majority response over total responses
    if len(response_counts) == 0:
        return None  # Handle cases with no responses
    majority_response_count = response_counts.max()
    repetitional_consistency_rate = majority_response_count / len(judge_responses)
    
    return repetitional_consistency_rate


if __name__=="__main__":
    MTBench_data_path = 'judgment_data/data_MTBench_repetition'
    DevBench_data_path = 'judgment_data/data_DevBench_repetition'

    MTBench_data = load_repetition_data(MTBench_data_path, nopt=3)
    DevBench_data = load_repetition_data(DevBench_data_path, nopt=2)

    MTBench_data = extract_answer(MTBench_data, benchmark='MTBench')
    DevBench_data = extract_answer(DevBench_data, benchmark='DevBench')
    
    calculate_repetition_bias_for_judge(MTBench_data, benchmark='MTBench', save_to_directory='MTBench_analysis/repetitional_consistency')
    calculate_repetition_bias_for_judge(DevBench_data, benchmark='DevBench', save_to_directory='DevBench_analysis/repetitional_consistency')