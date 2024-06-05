import pandas as pd
from collections import defaultdict

def calculate_length_stats(data, benchmark, reference_model):
    """
    Calculates the averate input length, output length, prompt length, and judgment length for each (Judge, Model, Task) unit.
    For multi-run Q&A datasets like MTBench, lengths are calculated as sum for each evaluation instance.

    ===========================================IMPORTANT=======================================================
        1. The input/output length refers to the task/question input/output length
        2. The prompt length is input to the Judge, so for price calculation, this is the "input length"
        3. The judgment length is the output of the Judge, so for price calculation, this is the "output length"
    ===========================================================================================================

    Parameters:
    - data (pd.DataFrame): A DataFrame with aggregated evaluations including 'cmp_index', judge columns,
    question columns, and answer columns.
    - benchmark: Benchmark name (e.g., MTBench or DevBench). This suggests which build_prompt method to use.
    - reference_model (str): The model to be used as the reference for comparison.
    
    Assumptions:
    - 'cmp_index' format: '{task};{model A};{model B}'
    - Question columns start with 'question', answer columns start with 'answer', and judgment columns start with 'judge_'

    Returns:
    - A DataFrame with input length, output length, prompt length, and judgment length for each (Judge, Model, Task).
    """
    # Initialize data structures for lengths
    input_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    output_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prompt_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    judgment_lengths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Extract question and answer columns
    question_columns = [col for col in data.columns if col.startswith('question')]
    answer_columns = [col for col in data.columns if col.startswith('answer')]

    # Initialize sets to store unique judges, models and tasks
    available_judges = set()
    available_models = set()
    available_tasks = set()

    # visited rows (swapped, so that visited before)
    visited_cmp_index = []

    # Calculate input and output lengths
    for _, row in data.iterrows():
        if row['cmp_index'] in visited_cmp_index:
            continue
        
        cmp_index_parts = row['cmp_index'].split(';')
        model_a, model_b = cmp_index_parts[1], cmp_index_parts[2]
        
        # Find the corresponding swapped row
        swapped_cmp_index = f"{cmp_index_parts[0]};{cmp_index_parts[2]};{cmp_index_parts[1]}"
        visited_cmp_index.append(swapped_cmp_index)
        
        # Determine if model A or model B is the model of interest (not the reference model)
        model_of_interest = model_b if model_a.lower() == reference_model else model_a
        available_models.add(model_of_interest)
        
        # Determine the task
        task = row['task']
        available_tasks.add(task)
        
        # Process each judge's input and output lengths
        for judge_col in [col for col in data.columns if col.startswith('extracted_')]:
            judge = judge_col[len('extracted_'):]  # Extract judge name
            available_judges.add(judge)
            
            # Calculate task input length (sum)
            input_length = row[question_columns].apply(lambda x: len(str(x))).sum()
            
            # Calculate output length (sum)
            output_length = row[answer_columns].apply(lambda x: len(str(x))).sum()
            
            # Calculate prompt length
            if benchmark == "MTBench":
                from subeval.subjective.prompt.MTBench_prompts import build_prompt
            elif benchmark == "DevBench":
                from subeval.subjective.prompt.DevBench_prompts import build_prompt
            else:
                raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
            prompt = build_prompt(row)
            prompt_length = len(prompt)
            
            # Calculate judgment length
            judgment_col = f"judge_{judge}"
            judgment_length = len(str(row[judgment_col]))
            
            # Store lengths for the model of interest
            input_lengths[judge][model_of_interest][task].append(input_length)
            output_lengths[judge][model_of_interest][task].append(output_length)
            prompt_lengths[judge][model_of_interest][task].append(prompt_length)
            judgment_lengths[judge][model_of_interest][task].append(judgment_length)

    # Compute average input, output, prompt, and judgment lengths
    avg_input_lengths = {}
    avg_output_lengths = {}
    avg_prompt_lengths = {}
    avg_judgment_lengths = {}

    for judge in available_judges:
        for model in available_models:
            for task in available_tasks:
                avg_input_length = pd.Series(input_lengths[judge][model][task]).mean() if input_lengths[judge][model][task] else 0
                avg_output_length = pd.Series(output_lengths[judge][model][task]).mean() if output_lengths[judge][model][task] else 0
                avg_prompt_length = pd.Series(prompt_lengths[judge][model][task]).mean() if prompt_lengths[judge][model][task] else 0
                avg_judgment_length = pd.Series(judgment_lengths[judge][model][task]).mean() if judgment_lengths[judge][model][task] else 0
                
                avg_input_lengths[(judge, model, task)] = avg_input_length
                avg_output_lengths[(judge, model, task)] = avg_output_length
                avg_prompt_lengths[(judge, model, task)] = avg_prompt_length
                avg_judgment_lengths[(judge, model, task)] = avg_judgment_length

    # Prepare data for DataFrame construction
    data_for_df = []
    for key in avg_input_lengths.keys():  # Keys are (judge, model, task)
        judge, model, task = key
        data_for_df.append({
            'Judge': judge,
            'Model': model,
            'Task': task,
            'avg_task_input_length': avg_input_lengths[key],
            'avg_task_output_length': avg_output_lengths[key],
            'avg_prompt_length': avg_prompt_lengths[key],
            'avg_judgment_length': avg_judgment_lengths[key]
        })

    # Convert to DataFrame
    length_results_df = pd.DataFrame(data_for_df)
    length_results_df.sort_values(by=['Judge', 'Model', 'Task'], inplace=True)

    print(f"{benchmark} Lengths calculation complete.")
    return length_results_df

def augment_length_stats(previous_results_df, length_results_df):
    """
    Augments the previous results DataFrame with the input length, output length, prompt length, and judgment length for each (Judge, Model, Task) unit.
    
    Parameters:
    - previous_results_df (pd.DataFrame): The previous results DataFrame obtained after calculating positional consistency and preference scores.
    - length_results_df (pd.DataFrame): The DataFrame with length stats obtained from `calculate_length_stats`.
    
    Returns:
    - A new DataFrame with the input length, output length, prompt length, and judgment length added to the previous results DataFrame.
    """
    # Create NaN columns at the end of previous_results_df named 'avg_task_input_length', 'avg_task_output_length', 'avg_prompt_length', and 'avg_judgment_length'
    previous_results_df['avg_task_input_length'] = float('nan')
    previous_results_df['avg_task_output_length'] = float('nan')
    previous_results_df['avg_prompt_length'] = float('nan')
    previous_results_df['avg_judgment_length'] = float('nan')
    
    # Iterate over each row of previous_results_df
    for index, row in previous_results_df.iterrows():
        judge = row['Judge']
        model = row['Model']
        task = row['Task']
        
        # Look for the corresponding (Judge, Model, Task) unit in length_results_df
        matching_row = length_results_df[
            (length_results_df['Judge'] == judge) &
            (length_results_df['Model'] == model) &
            (length_results_df['Task'] == task)
        ]
        
        # Fill the input, output, prompt, and judgment lengths to the last columns of previous_results_df if a match is found
        if not matching_row.empty:
            previous_results_df.at[index, 'avg_task_input_length'] = matching_row['avg_task_input_length'].values[0]
            previous_results_df.at[index, 'avg_task_output_length'] = matching_row['avg_task_output_length'].values[0]
            previous_results_df.at[index, 'avg_prompt_length'] = matching_row['avg_prompt_length'].values[0]
            previous_results_df.at[index, 'avg_judgment_length'] = matching_row['avg_judgment_length'].values[0]
    
    print("Augment Length stats to results_df complete.")
    return previous_results_df

if __name__=="__main__":
    # You'll need to first aggregate data and extract answers.

    MTBench_data = pd.read_csv("path/to/aggregated_data.csv")
    MTBench_length_stats_results_df = calculate_length_stats(data=MTBench_data, benchmark='MTBench', reference_model='vicuna-13b-v1.3')
    MTBench_previous_results_df = pd.read_csv("MTBench/results_with_both_winrates.csv")
    MTBench_results_with_length_stats = augment_length_stats(previous_results_df=MTBench_previous_results_df, length_results_df=MTBench_length_stats_results_df)
    MTBench_results_with_length_stats.to_csv("MTBench/results_with_both_winrates_and_length_stats.csv",index=False)
    print("MTBench results with length stats saved to MTBench/results_with_both_winrates_and_length_stats.csv")

    DevBench_data = pd.read_csv("path/to/aggregated_data.csv")
    DevBench_length_stats_results_df = calculate_length_stats(data=DevBench_data, benchmark='DevBench', reference_model='human')
    DevBench_previous_results_df = pd.read_csv("DevBench/results_with_both_winrates.csv")
    DevBench_results_with_length_stats = augment_length_stats(previous_results_df=DevBench_previous_results_df, length_results_df=DevBench_length_stats_results_df)
    DevBench_results_with_length_stats.to_csv("DevBench/results_with_both_winrates_and_length_stats.csv",index=False)
    print("DevBench results with length stats saved to DevBench/results_with_both_winrates_and_length_stats.csv")