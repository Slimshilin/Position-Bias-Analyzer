'''
These functions calculate the Overall Win Rate of models over a reference model for each (judge, model, task) unit.
The Overall Win Rate is different from the Consistent Win Rate by considering "inconsistency-as-tie".

The win rate formula is: overall_winrate = (consistent_wins + tie/2) / total

The `calculate_overall_winrate` function calculates the overall win rates for each (judge, model, task) unit and returns a DataFrame with the results.
The `augment_results_with_overall_winrates` function takes the previous results DataFrame and the overall win rate results DataFrame,
and augments the previous results DataFrame with a new column for the overall win rates.

The `reference_model` parameter specifies the model to be used as the reference for comparison.
'''

from collections import defaultdict
import pandas as pd

def calculate_overall_winrate(data, reference_model):
    """
    Calculates the Overall Win Rate of models over reference model for each (judge, model, task) unit.
    Overall Win Rate is different than Consistent Win Rate by cosidering "inconsistency-as-tie".
    
    Win rate formula: overall_winrate = (consistent_wins + tie/2) / total
    
    Parameters:
    - data (pd.DataFrame): A DataFrame with aggregated evaluations including 'cmp_index', judge columns,
      and extracted answers.
      
    Assumptions:
    - 'cmp_index' format: '{task};{model A};{model B}'
    - Judge columns are prefixed with 'judge_' and extracted answer columns with 'extracted_'
    
    Returns:
    - A DataFrame with overall win rates for each (judge, model, task).
    """
    # Initialize data structures for counts and scores
    overall_winrate_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
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
        
        # Determine if model A or model B is the model of interest (The other one that is not the reference model)
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
                        overall_winrate_counts[judge][model_of_interest][task] += 2
                elif choice_pair in ('AC','CA','BC','CB','AA','BB','CC'):  # Inconsistent (tie) or 'CC'
                    overall_winrate_counts[judge][model_of_interest][task] += 1
                
                if 'N' not in choice_pair: # Increment only for both extraction successful cases
                    total_counts[judge][model_of_interest][task] += 2
    
    # Compute overall win rates based on counts
    overall_winrates = {}
    
    for judge in available_judges:
        for model in available_models:
            for task in available_tasks:
                wins = overall_winrate_counts[judge][model][task]
                total = total_counts[judge][model][task]
                
                # Calculate overall win rate
                overall_winrate = wins / total if total else 0
                
                overall_winrates[(judge, model, task)] = overall_winrate
    
    # Prepare data for DataFrame construction
    data_for_df = []
    for key in overall_winrates.keys():  # Keys are (judge, model, task)
        judge, model, task = key
        data_for_df.append({
            'Judge': judge,
            'Model': model,
            'Task': task,
            'overall_winrate': overall_winrates[key]
        })
    
    # Convert to DataFrame
    overall_winrate_results_df = pd.DataFrame(data_for_df)
    overall_winrate_results_df.sort_values(by=['Judge', 'Model', 'Task'], inplace=True)
    
    print(f"Overall Win Rate calculation with reference model: {reference_model} complete.")
    return overall_winrate_results_df


def augment_results_with_overall_winrates(previous_results_df, overall_winrate_results_df):
    """
    Augments the previous results DataFrame with the overall win rates for each (Judge, Model, Task) unit.
    Call this function right after `calculate_overall_winrate`.
    
    Parameters:
    - previous_results_df (pd.DataFrame): The previous results DataFrame obtained after calculating positional consistency and preference scores.
    - overall_winrate_results_df (pd.DataFrame): The DataFrame with overall win rates obtained from `calculate_overall_winrate`.
      
    Returns:
    - A new DataFrame with the overall win rates added to the previous results DataFrame.
    """
    # Create an NaN column at the end of previous_results_df named 'overall_winrate'
    previous_results_df['overall_winrate'] = float('nan')
    
    # Iterate over each row of previous_results_df
    for index, row in previous_results_df.iterrows():
        judge = row['Judge']
        model = row['Model']
        task = row['Task']
        
        # Look for the corresponding (Judge, Model, Task) unit in overall_winrate_results_df
        matching_row = overall_winrate_results_df[
            (overall_winrate_results_df['Judge'] == judge) &
            (overall_winrate_results_df['Model'] == model) &
            (overall_winrate_results_df['Task'] == task)
        ]
        
        # Fill the winrate to the last column of previous_results_df if a match is found
        if not matching_row.empty:
            previous_results_df.at[index, 'overall_winrate'] = matching_row['overall_winrate'].values[0]
    
    print("Augment Overall Win Rate to results_df complete.")
    return previous_results_df


if __name__=="__main__":
   # You'll need to first aggregate data and extract answers.

   MTBench_data = pd.read_csv('path/to/aggregated_data.csv')
   MTBench_results_df = pd.read_csv('MTBench/(Judge-Model-Task)_results.csv')
   MTBench_overall_winrate_df = calculate_overall_winrate(data=MTBench_data, reference_model='vicuna-13b-v1.3')
   MTBench_result_with_overall_winrates = augment_results_with_overall_winrates(MTBench_results_df, MTBench_overall_winrate_df)
   MTBench_result_with_overall_winrates.to_csv(f"MTBench/results_with_overall-winrate.csv",index=False)
   print(f"MTBench results with overall win rates saved to MTBench/results_with_overall-winrate.csv")

   DevBench_data = pd.read_csv('path/to/aggregated_data.csv')
   DevBench_results_df = pd.read_csv('DevBench/(Judge-Model-Task)_results.csv')
   DevBench_overall_winrate_df = calculate_overall_winrate(data=DevBench_data, reference_model='human')
   DevBench_result_with_overall_winrates = augment_results_with_overall_winrates(DevBench_results_df, DevBench_overall_winrate_df)
   DevBench_result_with_overall_winrates.to_csv(f"DevBench/results_with_overall-winrate.csv",index=False)
   print(f"DevBench results with overall win rates saved to DevBench/results_with_overall-winrate.csv")