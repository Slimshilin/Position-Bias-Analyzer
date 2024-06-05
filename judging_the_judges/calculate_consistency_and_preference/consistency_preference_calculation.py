from collections import defaultdict
import pandas as pd
import numpy as np

def calculate_positional_consistency_and_preference_score(data, reference_model):
    """
    Calculates positional consistency and positional preference scores across judges, models, and tasks, and computes average
    consistency and preference scores along with the standard deviation for each judge.
    Since the judgment choice extraction is not all 100% from observation, we further calculates the extraction successful rate
    for each (Judge, Model, Task) unit and for each judge. Hence (1 - extraction successful rate) is the error rate.
    
    Formulas:
    - Positional Consistency = # of consistent pairs of judgment / # of total judgment pairs
    - primacy(recency) count = # of inconsistent judgments that prefer primacy(recency)
    - primacy(recency) percentage =  # of inconsistent judgpments that prefer primacy(recency) / # of total inconsistent judgment pairs
    - raw positional preference score = recency_count * recency_percentage - primacy_count * primacy_percentage,
        where primacy_percentage + recency_percentage = 1
    - Posisitional Preference Score:
        - Rescale raw positional preference score, mapping to [-1,1] and 0 remains 0
        - Rescale using the min possible and max possible score for each (Judge Task) unit, which is when positional consistency = 0 and all preference biased on one side
        - Assumption: the number of Models is same among all Judges and all Tasks, but the number of data points (evaluation instances) can be varied

    Parameters:
    - data (pd.DataFrame): A DataFrame with aggregated evaluations including 'cmp_index', judge columns,
      and extracted answers.
    - reference_model: the chosen reference/baseline model for comparision when running LLM-as-a-Judge.
        In other words, all evaluation instances should have one of the models being the reference_model.
        In our experiments, we choose vicuna-13b-v1.3 and human for MTBench and DevBench respectively.
      
    Assumptions:
    - Run this function AFTER aggregating judgments using the function `aggregate_judge_data` and `extract_answer` in `consistency_preference_util.py`
    - 'cmp_index' format: '{task};{model A};{model B}'
    - Judge columns are prefixed with 'judge_' and extracted answer columns with 'extracted_'
    
    Returns:
    - A DataFrame with detailed scores for each (Judge, Model, Task)
    - A DataFrame with average scores and standard deviation of per judge.
    """
    
    # Initialize data structures for counts and scores
    positional_consistency_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    primacy_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    recency_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    extraction_success_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    extraction_failure_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    judge_task_counts = defaultdict(lambda: defaultdict(int))
    total_judge_counts = defaultdict(int)
    
    # Additional dictionaries for overall judge scores
    total_primacy_counts = defaultdict(int)
    total_recency_counts = defaultdict(int)
    total_positional_consistency_counts = defaultdict(int)
    total_extraction_success_counts = defaultdict(int)
    total_extraction_failure_counts = defaultdict(int)
    
    # Additional dictionary for judge-level positional consistency and extraction successful rate
    judge_positional_consistency = defaultdict(list)
    judge_extraction_successful_rates = defaultdict(list)
    
    # Extract judge columns and identify models and tasks
    judge_columns = [col for col in data.columns if col.startswith('extracted_')]
    
    # visited rows (swapped, so that visited before)
    visited_cmp_index = []
    
    # Initialize sets to store unique judges, models and tasks
    available_judges = set()
    available_models = set()
    available_tasks = set()
    
    # Counting primacy-preferred, recency-preferred, and consistent judgment pairs
    for _, row in data.iterrows():
        if row['cmp_index'] in visited_cmp_index:
            continue
        
        cmp_index_parts = row['cmp_index'].split(';')
        model_a, model_b = cmp_index_parts[1], cmp_index_parts[2]
        
        # Find the corresponding swapped row
        swapped_cmp_index = f"{cmp_index_parts[0]};{cmp_index_parts[2]};{cmp_index_parts[1]}"
        swapped_row = data[data['cmp_index'] == swapped_cmp_index].iloc[0] if not data[data['cmp_index'] == swapped_cmp_index].empty else None
        visited_cmp_index.append(swapped_cmp_index)
        
        # Determine if model A or model B is the model of interest (i.e., the other model but NOT  the reference model)
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
            
                # Increment the task count for the current judge and task
                judge_task_counts[judge][task] += 1
                
                # If 'N' is extracted, by definition, this implies an extraction failure
                if choice =='N':
                    extraction_failure_counts[judge][model_of_interest][task] += 1
                    total_extraction_failure_counts[judge] += 1
                if swapped_choice == 'N':
                    extraction_failure_counts[judge][model_of_interest][task] += 1
                    total_extraction_failure_counts[judge] += 1
                
                # Choice pair
                choice_pair = choice + swapped_choice
                
                # If no 'N' extracted, then both choices are extracted successfully
                if 'N' not in choice_pair:
                    extraction_success_counts[judge][model_of_interest][task] += 2
                    total_extraction_success_counts[judge] += 2
                
                # For other valid choice pairs
                if choice_pair in ('AA', 'AC', 'CA'): # primacy-preferred
                    primacy_counts[judge][model_of_interest][task] += 1
                    total_primacy_counts[judge] += 1
                    total_judge_counts[judge] += 1
                elif choice_pair in ('BB', 'BC', 'CB'): # recency-preferred
                    recency_counts[judge][model_of_interest][task] += 1
                    total_recency_counts[judge] += 1
                    total_judge_counts[judge] += 1
                elif choice_pair in ('AB','BA','CC'):  # Consistent
                    positional_consistency_counts[judge][model_of_interest][task] += 1
                    total_positional_consistency_counts[judge] += 1
                    total_judge_counts[judge] += 1
                
            else: # No swapped row
                print(f"{judge}: {swapped_cmp_index} missing swapped evaluation extraction")
            
            
    # Compute positional consistency, preference scores, and extraction successful rate based on counts
    positional_consistency, primacy_percentages, recency_percentages, extraction_successful_rates = {}, {}, {}, {}
    raw_preference_scores, positional_preference_scores = {}, {}
    
    for judge in available_judges:
        for model in available_models:
            for task in available_tasks:
                ### Positional Consistency ###
                primacy = primacy_counts[judge][model][task]
                recency = recency_counts[judge][model][task]
                consistent = positional_consistency_counts[judge][model][task]
                total = primacy + recency + consistent
                
                # Calculate scores
                positional_consistency_value = consistent / total if total else 0
                primacy_percentage = primacy / (primacy + recency) if primacy + recency else 0
                recency_percentage = recency / (primacy + recency) if primacy + recency else 0
                raw_preference_score = recency * recency_percentage - primacy * primacy_percentage
                
                positional_consistency[(judge, model, task)] = positional_consistency_value
                primacy_percentages[(judge, model, task)] = primacy_percentage
                recency_percentages[(judge, model, task)] = recency_percentage
                raw_preference_scores[(judge, model, task)] = raw_preference_score
                
                # Store judge-level positional consistency
                judge_positional_consistency[judge].append(positional_consistency_value)
                
                
                ### Extraction Successful Rate ###
                success = extraction_success_counts[judge][model][task]
                failure = extraction_failure_counts[judge][model][task]
                total = success + failure
                
                # Calculate extraction_successful_rate
                extraction_successful_rate = success / total if total else 0
                extraction_successful_rates[(judge, model, task)] = extraction_successful_rate
                
                # Store juddge-level extraction successful rates
                judge_extraction_successful_rates[judge].append(extraction_successful_rate)
        
    # Rescale raw positional preference scores for each (Judge, Task) unit
    for judge in available_judges:
        for task in available_tasks:
            task_count = judge_task_counts[judge][task] / len(available_models) # Assume the number of Models is same among all Judges and Tasks
            min_possible_score = -task_count
            max_possible_score = task_count
            
            for model in available_models:
                raw_preference_score = raw_preference_scores[(judge, model, task)]
                if min_possible_score == max_possible_score:
                    positional_preference_score = 0
                else:
                    positional_preference_score = (raw_preference_score - min_possible_score) / (max_possible_score - min_possible_score) * 2 - 1
                
                positional_preference_scores[(judge, model, task)] = positional_preference_score

                    
    # Prepare data for DataFrame construction
    data_for_df = []
    for key in positional_consistency.keys():  # Keys are (judge, model, task)
        judge, model, task = key  # Unpack the key tuple
        data_for_df.append({
            'Judge': judge,
            'Model': model,
            'Task': task,
            'Positional Consistency': positional_consistency[key],
            'Primacy Count': primacy_counts[judge][model][task],
            'Recency Count': recency_counts[judge][model][task],
            'Primacy Percentage': primacy_percentages[key],
            'Recency Percentage': recency_percentages[key],
            'Raw Positional Preference Score': raw_preference_scores[key],
            'Positional Preference Score': positional_preference_scores[key],
            'Extraction Successful Rate': extraction_successful_rates[key]
        })
        
    # Convert to DataFrame
    results_df = pd.DataFrame(data_for_df)
    results_df.sort_values(by=['Judge', 'Model', 'Task'], inplace=True)
    
    print("(Judge, Model, Task) unit Positional Consistency and Preference Scores Calculation Complete.")

    # Compute average scores and standard deviation for each judge
    judge_averages = []
    for judge in available_judges:
        ### Positional consistency and preference scores ###
        total = total_judge_counts[judge]
        total_positional_consistency = total_positional_consistency_counts[judge] / total if total else 0
        std_positional_consistency = np.std(judge_positional_consistency[judge])
        avg_positional_preference_score = np.mean([score for key, score in positional_preference_scores.items() if key[0] == judge])
        std_positonal_preference_score = np.std([score for key, score in positional_preference_scores.items() if key[0] == judge])

        ### Extraction Successful Rate ###
        success = total_extraction_success_counts[judge]
        total = total_extraction_success_counts[judge] + total_extraction_failure_counts[judge]
        total_extraction_successful_rate = success / total if total else 0
        std_extraction_successful_rate = np.std(judge_extraction_successful_rates[judge])
        
        judge_averages.append({
            'Judge': judge,
            'Average Positional Consistency': total_positional_consistency,
            'Positional Consistency Std': std_positional_consistency,
            'Average Positional Preference Score': avg_positional_preference_score,
            'Positional Preference Score Std': std_positonal_preference_score,
            'Average Extraction Successful Rate': total_extraction_successful_rate,
            'Extraction Successful Rate Std': std_extraction_successful_rate
        })
    
    averages_df = pd.DataFrame(judge_averages)
    averages_df.sort_values(by='Judge', inplace=True)

    print("Judge Average Positional Consistency and Preference Scores Calculation Complete.")

    return results_df, averages_df


if __name__=="__main__":
    # You'll need to first aggregate data and extract answers.

    # MTBench
    save_to_path = 'MTBench'
    data = pd.read_csv('path/to/aggregated_data.csv')
    results_df, averages_df = calculate_positional_consistency_and_preference_score(data, reference_model="vicuna-13b-v1.3")
    results_df.to_csv(f"{save_to_path}/(Judge-Model-Task)_results.csv",index=False)
    averages_df.to_csv(f"{save_to_path}/Judge_average_results.csv",index=False)

    # DevBench
    save_to_path = 'DevBench'
    data = pd.read_csv('path/to/aggregated_data.csv')
    results_df, averages_df = calculate_positional_consistency_and_preference_score(data, reference_model="human")
    results_df.to_csv(f"{save_to_path}/(Judge-Model-Task)_results.csv",index=False)
    averages_df.to_csv(f"{save_to_path}/Judge_average_results.csv",index=False)