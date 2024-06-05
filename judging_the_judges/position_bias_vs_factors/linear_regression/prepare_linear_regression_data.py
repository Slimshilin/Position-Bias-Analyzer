'''
This function prepares the dataframe to run a linear regression cosidering all factors.
It mainly augmented the context window, max output length, and familial properties.

Call this function after calculating all the metrics:
    - positional consistency and preference score
    - win rates
    - length stats

Although it serves for `benchmark` like MTBench or DevBench, it can also be used for combined datasets.
You may combine the datasets (after metrics calculated) and then call this preparation function,
or you may also call this function seperately for each dataset and then combine manually/using helper functions.

'''


import pandas as pd
import numpy as np
import os

context_window = {
    'gpt-4-0125-preview': 128000,
    'gpt-4-1106-preview': 128000,
    'gpt-4-0613': 8192,
    'gpt-3.5-turbo-0125': 16385,
    'gpt-3.5-turbo-1106': 16385,
    'claude-3-sonnet-20240229': 200000,
    'claude-3-haiku-20240307': 200000,
    'claude-3-opus-20240229': 200000,
    'gemini-pro': 30720
}

max_output_length = {
    'gpt-4-0125-preview': 4096,
    'gpt-4-1106-preview': 4096,
    'gpt-4-0613': 4096,
    'gpt-3.5-turbo-0125': 4096,
    'gpt-3.5-turbo-1106': 4096,
    'claude-3-sonnet-20240229': 4096,
    'claude-3-haiku-20240307': 4096,
    'claude-3-opus-20240229': 4096,
    'gemini-pro': 2048
}

family_map = {
    'gpt-4-0125-preview': 'GPT-4 Turbo',
    'gpt-4-1106-preview': 'GPT-4 Turbo',
    'gpt-4-0613': 'GPT-4',
    'gpt-3.5-turbo-0125': 'GPT-3.5',
    'gpt-3.5-turbo-1106': 'GPT-3.5',
    'claude-3-sonnet-20240229': 'Claude-3',
    'claude-3-haiku-20240307': 'Claude-3',
    'claude-3-opus-20240229': 'Claude-3',
    'gemini-pro': 'Gemini'
}

def prepare_linear_regression_data(results_df, benchmark, context_window, max_output_length, family_map, save_to_directory=None):
    """
    Prepares data for linear regression by integrating context window, max output length, and family information.
    
    Parameters:
    - results_df: DataFrame containing Judge, Model, Task, and various metrics.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the file names.
    - context_window: Dictionary mapping Judges to their context window sizes.
    - max_output_length: Dictionary mapping Judges to their maximum output lengths.
    - family_map: Dictionary mapping Judges to their corresponding model family.
    - save_to_directory: Optional; if provided, saves the resulting DataFrame to this directory.
    
    Returns:
    - DataFrame ready for linear regression, with original and dummy variables for Judge, Model, Task, and family.
    """
    # Select relevant columns and make a copy to avoid SettingWithCopyWarning
    data_for_model = results_df[[
        'Judge', 'Model', 'Task', 
        'Positional Consistency', 'overall_winrate', 'Positional Preference Score',
        'avg_task_input_length', 'avg_task_output_length', 
        'avg_prompt_length'
    ]].copy()
    
    # Quantify quality gap
    data_for_model['quality_gap'] = abs(data_for_model['overall_winrate'] - 0.5)
    
    # Calculate the log of length stats
    data_for_model['log_task_input_length'] = np.log(data_for_model['avg_task_input_length'])
    data_for_model['log_task_output_length'] = np.log(data_for_model['avg_task_output_length'])
    data_for_model['log_prompt_length'] = np.log(data_for_model['avg_prompt_length'])

    # Map additional attributes to the DataFrame using .loc for safe in-place modification
    data_for_model.loc[:, 'context_window'] = data_for_model['Judge'].map(context_window)
    data_for_model.loc[:, 'max_output_length'] = data_for_model['Judge'].map(max_output_length)
    data_for_model.loc[:, 'family'] = data_for_model['Judge'].map(family_map)

    # Rename columns for clarity and consistency
    column_renames = {
        'avg_task_input_length': 'task_input_length',
        'avg_task_output_length': 'task_output_length',
        'avg_prompt_length': 'prompt_length'
    }
    data_for_model.rename(columns=column_renames, inplace=True)

    # Create dummy variables for categorical columns, ensuring to keep the original columns
    categorical_columns = ['Judge', 'Task', 'family']
    dummies = pd.get_dummies(data_for_model[categorical_columns], prefix=categorical_columns, prefix_sep='_')
    prepared_data = pd.concat([data_for_model, dummies], axis=1)

    # Optionally save the prepared DataFrame to a specified directory
    if save_to_directory is not None:
        os.makedirs(save_to_directory, exist_ok=True)
        file_name = f"{benchmark}_linear_regression_data.csv"
        file_path = os.path.join(save_to_directory, file_name)
        prepared_data.to_csv(file_path, index=False)
        print(f"{benchmark} linear regression data saved to {file_path}")
    
    return prepared_data


if __name__=="__main__":
    MTBench_results_df = pd.read_csv('MTBench/results_with_both_winrates_and_length_stats.csv')
    MTBench_prepared_data = prepare_linear_regression_data(results_df=MTBench_results_df,
                                                           benchmark='MTBench',
                                                           context_window=context_window,
                                                           max_output_length=max_output_length,
                                                           family_map=family_map,
                                                           save_to_directory='MTBench/linear_regression')
    
    DevBench_results_df = pd.read_csv('DevBench/results_with_both_winrates_and_length_stats.csv')
    DevBench_prepared_data = prepare_linear_regression_data(results_df=DevBench_results_df,
                                                            benchmark='DevBench',
                                                            context_window=context_window,
                                                            max_output_length=max_output_length,
                                                            family_map=family_map,
                                                            save_to_directory='DevBench/linear_regression')