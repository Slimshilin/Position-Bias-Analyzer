'''
These functions calculate the Consistent Win Rate of models over a reference model for each (judge, model, task) unit.
The Consistent Win Rate excludes the inconsistent judgment pairs and only calculates the win rates for the consistent judgment pairs.

The win rate formula is: consistent_winrate = (consistent wins + (consistent tie/2)) / consistent total

The `calculate_consistent_winrate` function calculates the consistent win rates for each (judge, model, task) unit and returns a DataFrame with the results.
The `augment_results_with_consistent_winrates` function takes the previous results DataFrame and the consistent win rate results DataFrame,
and augments the previous results DataFrame with a new column for the consistent win rates.

The `reference_model` parameter specifies the model to be used as the reference for comparison.
'''

import pandas as pd
from collections import defaultdict

def calculate_consistent_winrate(data, reference_model):
    """
    Calculates the consistent win rate of models over reference model for each (judge, model, task) unit.
    Win rate formula: consistent_winrate = (consistent wins + (consistent tie/2)) / consistent total

    Parameters:
    - data (pd.DataFrame): A DataFrame with aggregated evaluations including 'cmp_index', judge columns,
    and extracted answers.
    - reference_model (str): The model to be used as the reference for comparison.
    
    Assumptions:
    - 'cmp_index' format: '{task};{model A};{model B}'
    - Judge columns are prefixed with 'judge_' and extracted answer columns with 'extracted_'

    Returns:
    - A DataFrame with consistent win rates for each (judge, model, task).
    """
    # Initialize data structures for counts and scores
    consistent_winrate_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Extract judge columns and identify models and tasks
    judge_columns = [col for col in data.columns if col.startswith('extracted_')]

    # visited rows (swapped, so that visited before)
    visited_cmp_index = []

    # Initialize sets to store unique judges, models and tasks
    available_judges = set()
    available_models = set()
    available_tasks = set()

    # Counting wins and total
    for _, row in data.iterrows():
        if row['cmp_index'] in visited_cmp_index:
            continue
        
        cmp_index_parts = row['cmp_index'].split(';')
        model_a, model_b = cmp_index_parts[1], cmp_index_parts[2]
        
        # Find the corresponding swapped row
        swapped_cmp_index = f"{cmp_index_parts[0]};{cmp_index_parts[2]};{cmp_index_parts[1]}"
        swapped_row = data[data['cmp_index'] == swapped_cmp_index].iloc[0] if not data[data['cmp_index'] == swapped_cmp_index].empty else None
        visited_cmp_index.append(swapped_cmp_index)
        
        # Determine if model A or model B is the model of interest (not the reference model)
        model_of_interest = model_b if model_a.lower() == reference_model else model_a
        available_models.add(model_of_interest)
        
        # Determine the task
        task = row['task']
        available_tasks.add(task)
        
        # Process each judge's decision
        for judge_col in judge_columns:
            judge = judge_col[len('extracted_'):]  # Extract judge name
            available_judges.add(judge)
            
            # Extract choices
            choice = row[judge_col]
            if swapped_row is not None and judge_col in swapped_row:
                swapped_choice = swapped_row[judge_col]
                
                # Choice pair
                choice_pair = choice + swapped_choice
                                
                if choice_pair in ('AB', 'BA'):  # Consistent and only choice
                    if choice == 'A' and model_a == model_of_interest or choice == 'B' and model_b == model_of_interest:
                        consistent_winrate_counts[judge][model_of_interest][task] += 2
                    total_counts[judge][model_of_interest][task] += 2
                elif choice_pair == 'CC':  # CC
                    consistent_winrate_counts[judge][model_of_interest][task] += 1
                    total_counts[judge][model_of_interest][task] += 2

    # Compute consistent win rates based on counts
    consistent_winrates = {}

    for judge in available_judges:
        for model in available_models:
            for task in available_tasks:
                wins = consistent_winrate_counts[judge][model][task]
                total = total_counts[judge][model][task]
                
                # Calculate consistent win rate
                consistent_winrate = wins / total if total else 0
                
                consistent_winrates[(judge, model, task)] = consistent_winrate

    # Prepare data for DataFrame construction
    data_for_df = []
    for key in consistent_winrates.keys():  # Keys are (judge, model, task)
        judge, model, task = key
        data_for_df.append({
            'Judge': judge,
            'Model': model,
            'Task': task,
            'consistent_winrate': consistent_winrates[key]
        })

    # Convert to DataFrame
    consistent_winrate_results_df = pd.DataFrame(data_for_df)
    consistent_winrate_results_df.sort_values(by=['Judge', 'Model', 'Task'], inplace=True)

    print(f"Consistent Win Rate calculation with reference model: {reference_model} complete.")
    return consistent_winrate_results_df


def augment_results_with_consistent_winrates(previous_results_df, consistent_winrate_results_df):
    """
    Augments the previous results DataFrame with the consistent win rates for each (Judge, Model, Task) unit.

    Parameters:
    - previous_results_df (pd.DataFrame): The previous results DataFrame obtained after calculating positional consistency and preference scores.
    - consistent_winrate_results_df (pd.DataFrame): The DataFrame with consistent win rates obtained from `calculate_consistent_winrate`.

    Returns:
    - A new DataFrame with the consistent win rates added to the previous results DataFrame.
    """
    # Create an NaN column at the end of previous_results_df named 'consistent_winrate'
    previous_results_df['consistent_winrate'] = float('nan')

    # Iterate over each row of previous_results_df
    for index, row in previous_results_df.iterrows():
        judge = row['Judge']
        model = row['Model']
        task = row['Task']

        # Look for the corresponding (Judge, Model, Task) unit in consistent_winrate_results_df
        matching_row = consistent_winrate_results_df[
            (consistent_winrate_results_df['Judge'] == judge) &
            (consistent_winrate_results_df['Model'] == model) &
            (consistent_winrate_results_df['Task'] == task)
        ]

        # Fill the winrate to the last column of previous_results_df if a match is found
        if not matching_row.empty:
            previous_results_df.at[index, 'consistent_winrate'] = matching_row['consistent_winrate'].values[0]
    print("Augment Consistent Win Rate to results_df complete.")
    return previous_results_df


if __name__=="__main__":
   # You'll need to first aggregate data and extract answers.

   MTBench_data = pd.read_csv('path/to/aggregated_data.csv')
   MTBench_results_df = pd.read_csv('MTBench/results_with_overall-winrate.csv')
   MTBench_consistent_winrate_df = calculate_consistent_winrate(data=MTBench_data, reference_model='vicuna-13b-v1.3')
   MTBench_result_with_consistent_winrates = augment_results_with_consistent_winrates(MTBench_results_df, MTBench_consistent_winrate_df)
   MTBench_result_with_consistent_winrates.to_csv(f"MTBench/results_with_both_winrates.csv",index=False)
   print(f"MTBench results with consistent win rates saved to MTBench/results_with_both_winrates.csv")

   DevBench_data = pd.read_csv('path/to/aggregated_data.csv')
   DevBench_results_df = pd.read_csv('DevBench/results_with_overall-winrate.csv')
   DevBench_consistent_winrate_df = calculate_consistent_winrate(data=DevBench_data, reference_model='human')
   DevBench_result_with_consistent_winrates = augment_results_with_consistent_winrates(DevBench_results_df, DevBench_consistent_winrate_df)
   DevBench_result_with_consistent_winrates.to_csv(f"DevBench/results_with_both_winrates.csv",index=False)
   print(f"DevBench results with consistent win rates saved to DevBench/results_with_both_winrates.csv")