'''
This function runs a linear regression considering all Judge, Model, and Task level impacting factors and dummy variables.
It servs for two purposes:
    1. See if the positional consistency or preference score is linearly predictable (turns out NO)
    2. Explore signifance of the impact of each factor on position bias

Here we support dummy variable included or discluded. The dummy variables are individual Judges and Tasks.
We do not consider Model because 
    a) that's a lot 
    b) that's kind of useless, lacking generalizability

You should call the preparation function first and then run the linear regression.
Similar to what's mentioned in the preparation function, you may either fun on each dataset or a combination of them.
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def run_linear_regression(data: pd.DataFrame, benchmark, dummy_included=True, save_to_directory=None):
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory, exist_ok=True)

    # Select relevant columns
    selected_columns = ['Positional Consistency', 'Positional Preference Score', 'quality_gap', 'log_task_input_length', 'log_task_output_length', 'log_prompt_length', 'context_window', 'max_output_length']
    if dummy_included:
        dummy_columns = [col for col in data.columns if col.startswith(('Judge_', 'Task_', 'family_'))]
        selected_columns.extend(dummy_columns)

    # Filter data
    data = data[selected_columns]

    # Handle missing values
    data = data.fillna(False)  # Fill with False (because only dummy variables could be NaN during concatenation)

    # Convert selected columns to numeric data types
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.astype(int)  # For dummy

    # Define the dependent variables 'y'
    y_consistency = data['Positional Consistency']
    y_preference = data['Positional Preference Score']

    # Define the independent variables 'X'
    X = data.drop(['Positional Consistency', 'Positional Preference Score'], axis=1)

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the linear regression models
    with np.errstate(divide='ignore', invalid='ignore'):
        model_consistency = sm.OLS(y_consistency, X).fit()
        model_preference = sm.OLS(y_preference, X).fit()

    # Get the summary statistics of the models
    with np.errstate(divide='ignore'):
        summary_consistency = model_consistency.summary()
        summary_preference = model_preference.summary()

    # Optionally, save the summary statistics to files if save_to_directory is not None
    dummy_status = "dummy_included" if dummy_included else "dummy_excluded"
    consistency_file_name = f"{benchmark}_predict_consistency_{dummy_status}.txt"
    preference_file_name = f"{benchmark}_predict_preference_{dummy_status}.txt"
    consistency_file_path = os.path.join(save_to_directory, consistency_file_name)
    preference_file_path = os.path.join(save_to_directory, preference_file_name)

    with open(consistency_file_path, 'w') as file:
        file.write(summary_consistency.as_text())
    with open(preference_file_path, 'w') as file:
        file.write(summary_preference.as_text())

    print(f"{benchmark} linear regression summary for positional consistency saved to {consistency_file_path}")
    print(f"{benchmark} linear regression summary for preference score saved to {preference_file_path}")

    return summary_consistency, summary_preference


if __name__ == "__main__":
    MTBench_data = pd.read_csv('MTBench/linear_regression/MTBench_linear_regression_data.csv')
    MTBench_summary_consistency, MTBench_summary_preference = run_linear_regression(data=MTBench_data,
                                                                                     benchmark='MTBench',
                                                                                     dummy_included=True,
                                                                                     save_to_directory="MTBench/linear_regression")
    print("MTBench Positional Consistency:")
    print(MTBench_summary_consistency)
    print("\nMTBench Preference Score:")
    print(MTBench_summary_preference)

    DevBench_data = pd.read_csv('DevBench/linear_regression/DevBench_linear_regression_data.csv')
    DevBench_summary_consistency, DevBench_summary_preference = run_linear_regression(data=DevBench_data,
                                                                                      benchmark='DevBench',
                                                                                      dummy_included=True,
                                                                                      save_to_directory="DevBench/linear_regression")
    print("DevBench Positional Consistency:")
    print(DevBench_summary_consistency)
    print("\nDevBench Preference Score:")
    print(DevBench_summary_preference)