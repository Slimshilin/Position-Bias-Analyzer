'''
These functions plot the relationship between the price and positional consistency of each judge.
The `plot_judge_task_level_price_vs_positional_consistency` function plots this relationship for each task,
while the `plot_judge_level_price_vs_positional_consistency` function plots the average across all tasks.

The price is calculated based on the input and output pricing maps, which specify the price per million tokens
for each judge. The length of the prompt and judgment are used to approximate the number of tokens.

So the functions should be called after length calculations.

The graphs use different colors and markers for each judge family (e.g., GPT-4, GPT-3.5, etc.), and the transparency
of the markers is adjusted based on the order of the models within each family.

Since the judge choices and number of tasks may vary, feel free to revise the functions according to specific needs.

The `benchmark` parameter specifies whether the graphs are for MTBench or DevBench.

The resulting graphs are saved to the specified `save_to_directory` if provided, otherwise they are displayed.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os

input_pricing_map = {
    'gpt-4-0125-preview': 10.00,
    'gpt-4-1106-preview': 10.00,
    'gpt-4-0613': 30.00,
    'gpt-3.5-turbo-0125': 0.50,
    'gpt-3.5-turbo-1106': 1.00,
    'claude-3-haiku-20240307': 0.25,
    'claude-3-sonnet-20240229': 3.00,
    'claude-3-opus-20240229': 15.00,
    'gemini-pro': 0.00
}

output_pricing_map = {
    'gpt-4-0125-preview': 30.00,
    'gpt-4-1106-preview': 30.00,
    'gpt-4-0613': 60.00,
    'gpt-3.5-turbo-0125': 1.50,
    'gpt-3.5-turbo-1106': 2.00,
    'claude-3-haiku-20240307': 1.25,
    'claude-3-sonnet-20240229': 15.00,
    'claude-3-opus-20240229': 75.00,
    'gemini-pro': 0.00
}

def plot_judge_task_level_price_vs_positional_consistency(results_df, benchmark, graph_title, subplot_layout, input_pricing_map, output_pricing_map, save_to_directory=None):
    """
    Plots the relationship between the price and positional consistency of each judge for a task.

    Parameters:
    - results_df (pd.DataFrame): DataFrame with columns [Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count, Primacy Percentage, Recency Percentage, Raw Positional Preference Score, Positional Preference Score, Extraction Successful Rate, consistent_winrate, tieConsidered_winrate, avg_task_input_length, avg_task_output_length, avg_prompt_length, avg_judgment_length].
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The overall title for the figure.
    - subplot_layout (tuple): The layout of subplots, e.g., (3, 3).
    - input_pricing_map (dict): A dictionary mapping judge names to prices per million tokens for input (prompt). Note that here we're using length to approximate the number of tokens.
    - output_pricing_map (dict): A dictionary mapping judge names to prices per million tokens for output (judgment). Note that here we're using length to approximate the number of tokens.
    - save_to_directory (str, optional): The directory where the figure will be saved. If None (default), the figure will be displayed without saving.

    Raises:
    - ValueError: If the number of tasks exceeds the limit of the subplot_layout.
    """
    tasks = results_df['Task'].unique()
    if len(tasks) > subplot_layout[0] * subplot_layout[1]:
        raise ValueError("The number of tasks exceeds the limit of the subplot_layout.")

    if benchmark == "MTBench":
        fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=(18, 16))
    elif benchmark == "DevBench":
        fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=(22, 16))
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
        
    fig.suptitle(graph_title, fontsize=20)

    for i, task in enumerate(tasks):
        if i >= subplot_layout[0] * subplot_layout[1]:
            break

        row = i // subplot_layout[1]
        col = i % subplot_layout[1]
        ax = axes[row, col]

        task_data = results_df[results_df['Task'] == task]
        # For unknown ones
        # judges = results_df['Judge'].unique()
        # But for the known 9 judges, specify the order of judges here
        judges=['gpt-4-0125-preview', 
                'gpt-4-1106-preview', 
                'gpt-4-0613', 
                'gpt-3.5-turbo-0125', 
                'gpt-3.5-turbo-1106', 
                'claude-3-opus-20240229', 
                'claude-3-sonnet-20240229', 
                'claude-3-haiku-20240307', 
                'gemini-pro']

        # Define a color map and marker map for different families
        family_color_map = {
            'claude-3': 'blue',
            'gemini-pro': 'green',
            'gpt-3.5': 'red',
            'gpt-4': 'purple'
        }
        family_marker_map = {
            'claude-3': 'o',
            'gemini-pro': 's',
            'gpt-3.5': '^',
            'gpt-4': 'D'
        }

        # Define the order of models within each family
        family_order = {
            'claude-3': ['opus-20240229', 'sonnet-20240229', 'haiku-20240307'],
            'gpt-3.5': ['turbo-0125', 'turbo-1106'],
            'gpt-4': ['0125-preview', '1106-preview', '0613'],
            'gemini-pro':[""]
        }

        # Create dictionaries to store the alpha values and model counts for each family
        family_alpha_map = {}
        family_model_count = {}

        # Create dictionaries to store the legend handles and labels for each family
        family_legend_handles = {}
        family_legend_labels = {}

        # Count the number of models in each family
        for judge in judges:
            family = '-'.join(judge.split('-')[:2])
            if family not in family_model_count:
                family_model_count[family] = 1
            else:
                family_model_count[family] += 1

        # Assign alpha values for each family based on the order
        for family in family_model_count:
            if family_model_count[family] == 1:
                family_alpha_map[family] = [1.0]
            else:
                family_alpha_map[family] = np.linspace(1.0, 0.2, family_model_count[family])
                
        for judge in judges:
            judge_task_data = task_data[task_data['Judge'] == judge]
            consistency_scores = judge_task_data['Positional Consistency'].values
            avg_prompt_lengths = judge_task_data['avg_prompt_length'].values
            avg_judgment_lengths = judge_task_data['avg_judgment_length'].values

            avg_consistency_score = np.mean(consistency_scores)
            avg_prompt_length = np.mean(avg_prompt_lengths)
            avg_judgment_length = np.mean(avg_judgment_lengths)

            input_price = avg_prompt_length / 1e6 * input_pricing_map[judge]
            output_price = avg_judgment_length / 1e6 * output_pricing_map[judge]
            total_price = input_price + output_price

            family = '-'.join(judge.split('-')[:2])
            color = family_color_map[family]
            marker = family_marker_map[family]

            # Get the alpha value for the current model based on its order within the family
            model_name = '-'.join(judge.split('-')[2:])
            model_index = family_order[family].index(model_name)
            alpha = family_alpha_map[family][model_index]

            # Convert color to RGBA format
            rgba_color = mcolors.to_rgba(color, alpha=alpha)

            # Create a scatter plot for the current judge
            handle = ax.scatter(avg_consistency_score, total_price, edgecolors=color, facecolors=rgba_color, marker=marker, linewidths=1.5)

            # Add the legend handle and label to the dictionaries for the current family
            if family not in family_legend_handles:
                family_legend_handles[family] = []
                family_legend_labels[family] = []
            family_legend_handles[family].append(handle)
            family_legend_labels[family].append(judge)

        ax.set_title(task, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Positional Consistency', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(color='black', alpha=0.2, linestyle='-', linewidth=0.5)

        # Create a list to store the legend handles and labels in the desired order
        legend_handles = []
        legend_labels = []

        # Iterate over the families in the desired order
        for family in family_color_map.keys():
            if family in family_legend_handles:
                legend_handles.extend(family_legend_handles[family])
                legend_labels.extend(family_legend_labels[family])

        # Create a single legend with the handles and labels in the desired order
        if benchmark == "MTBench":
            ax.legend(legend_handles, legend_labels, fontsize=12, loc='upper left')
        elif benchmark == "DevBench":
            ax.legend(legend_handles, legend_labels, fontsize=8, loc='upper left')

    for i in range(len(tasks), subplot_layout[0] * subplot_layout[1]):
        row = i // subplot_layout[1]
        col = i % subplot_layout[1]
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_to_directory is not None:
        os.makedirs(save_to_directory, exist_ok=True)
        plt.savefig(f"{save_to_directory}/{benchmark}_price_vs_positional_consistency_task-level.png")
        print(f"{benchmark} judge task level price vs positional consistency graph saved to {save_to_directory}/{benchmark}_price_vs_positional_consistency_task-level.png")
    else:
        plt.show()


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

def plot_judge_level_price_vs_positional_consistency(results_df, benchmark, graph_title, input_pricing_map, output_pricing_map, save_to_directory=None):
    """
    Plots the relationship between the price and positional consistency of each judge averaged across all tasks.

    Parameters:
    - results_df (pd.DataFrame): DataFrame with columns [Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count, Primacy Percentage, Recency Percentage, Raw Positional Preference Score, Positional Preference Score, Extraction Successful Rate, consistent_winrate, tieConsidered_winrate, avg_task_input_length, avg_task_output_length, avg_prompt_length, avg_judgment_length].
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The overall title for the figure.
    - input_pricing_map (dict): A dictionary mapping judge names to prices per million tokens for input (prompt). Note that here we're using length to approximate the number of tokens.
    - output_pricing_map (dict): A dictionary mapping judge names to prices per million tokens for output (judgment). Note that here we're using length to approximate the number of tokens.
    - save_to_directory (str, optional): The directory where the figure will be saved. If None (default), the figure will be displayed without saving.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(graph_title, fontsize=16)

    # For unknown ones
    # judges = results_df['Judge'].unique()
    # But for the known 9 judges, specify the order of judges here
    judges=['gpt-4-0125-preview', 
            'gpt-4-1106-preview', 
            'gpt-4-0613', 
            'gpt-3.5-turbo-0125', 
            'gpt-3.5-turbo-1106', 
            'claude-3-opus-20240229', 
            'claude-3-sonnet-20240229', 
            'claude-3-haiku-20240307', 
            'gemini-pro']


    # Define a color map and marker map for different families
    family_color_map = {
        'claude-3': 'blue',
        'gemini-pro': 'green',
        'gpt-3.5': 'red',
        'gpt-4': 'purple'
    }
    family_marker_map = {
        'claude-3': 'o',
        'gemini-pro': 's',
        'gpt-3.5': '^',
        'gpt-4': 'D'
    }

    # Define the order of models within each family
    family_order = {
        'claude-3': ['opus-20240229', 'sonnet-20240229', 'haiku-20240307'],
        'gpt-3.5': ['turbo-0125', 'turbo-1106'],
        'gpt-4': ['0125-preview', '1106-preview', '0613'],
        'gemini-pro':[""]
    }

    # Create dictionaries to store the alpha values and model counts for each family
    family_alpha_map = {}
    family_model_count = {}

    # Create dictionaries to store the legend handles and labels for each family
    family_legend_handles = {}
    family_legend_labels = {}

    # Count the number of models in each family
    for judge in judges:
        family = '-'.join(judge.split('-')[:2])
        if family not in family_model_count:
            family_model_count[family] = 1
        else:
            family_model_count[family] += 1

    # Assign alpha values for each family based on the order
    for family in family_model_count:
        if family_model_count[family] == 1:
            family_alpha_map[family] = [1.0]
        else:
            family_alpha_map[family] = np.linspace(1.0, 0.2, family_model_count[family])

    for judge in judges:
        judge_data = results_df[results_df['Judge'] == judge]
        consistency_scores = judge_data['Positional Consistency'].values
        avg_prompt_lengths = judge_data['avg_prompt_length'].values
        avg_judgment_lengths = judge_data['avg_judgment_length'].values

        avg_consistency_score = np.mean(consistency_scores)
        avg_prompt_length = np.mean(avg_prompt_lengths)
        avg_judgment_length = np.mean(avg_judgment_lengths)

        input_price = avg_prompt_length / 1e6 * input_pricing_map[judge]
        output_price = avg_judgment_length / 1e6 * output_pricing_map[judge]
        total_price = input_price + output_price

        family = '-'.join(judge.split('-')[:2])
        color = family_color_map[family]
        marker = family_marker_map[family]

        # Get the alpha value for the current model based on its order within the family
        model_name = '-'.join(judge.split('-')[2:])
        model_index = family_order[family].index(model_name)
        alpha = family_alpha_map[family][model_index]

        # Convert color to RGBA format
        rgba_color = mcolors.to_rgba(color, alpha=alpha)

        # Create a scatter plot for the current judge
        handle = ax.scatter(avg_consistency_score, total_price, edgecolors=color, facecolors=rgba_color, marker=marker, linewidths=1.5)

        # Add the legend handle and label to the dictionaries for the current family
        if family not in family_legend_handles:
            family_legend_handles[family] = []
            family_legend_labels[family] = []
        family_legend_handles[family].append(handle)
        family_legend_labels[family].append(judge)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Positional Consistency')
    ax.set_ylabel('Price')

    # Create a list to store the legend handles and labels in the desired order
    legend_handles = []
    legend_labels = []

    # Iterate over the families in the desired order
    for family in family_color_map.keys():
        if family in family_legend_handles:
            legend_handles.extend(family_legend_handles[family])
            legend_labels.extend(family_legend_labels[family])

    # Create a single legend with the handles and labels in the desired order
    ax.legend(legend_handles, legend_labels, fontsize=10)

    # Add grid lines with diluted black color
    ax.grid(color='black', alpha=0.2, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_to_directory is not None:
        os.makedirs(save_to_directory, exist_ok=True)
        plt.savefig(f"{save_to_directory}/{benchmark}_price_vs_positional_consistency_overall.png")
        print(f"{benchmark} judge level price vs positional consistency graph saved to {save_to_directory}/{benchmark}_price_vs_positional_consistency_overall.png")
    else:
        plt.show()

if __name__=="__main__":
    MTBench_results_df = pd.read_csv("MTBench/results_with_both_winrates_and_length_stats.csv")
    DevBench_results_df = pd.read_csv("DevBench/results_with_both_winrates_and_length_stats.csv")

    plot_judge_task_level_price_vs_positional_consistency(results_df=MTBench_results_df,
                                                        benchmark='MTBench',
                                                        graph_title='MTBench Price vs. Positional Consistency (Task Level)',
                                                        subplot_layout=(4,2),
                                                        input_pricing_map=input_pricing_map,
                                                        output_pricing_map=output_pricing_map,
                                                        save_to_directory="MTBench/price_vs_consistency")
    plot_judge_level_price_vs_positional_consistency(results_df=MTBench_results_df,
                                                    benchmark='MTBench',
                                                    graph_title='MTBench Price vs. Positional Consistency',
                                                    input_pricing_map=input_pricing_map,
                                                    output_pricing_map=output_pricing_map,
                                                    save_to_directory="MTBench/price_vs_consistency")

    plot_judge_task_level_price_vs_positional_consistency(results_df=DevBench_results_df,
                                                        benchmark='DevBench',
                                                        graph_title='DevBench Price vs. Positional Consistency (Task Level)',
                                                        subplot_layout=(5,3),
                                                        input_pricing_map=input_pricing_map,
                                                        output_pricing_map=output_pricing_map,
                                                        save_to_directory="DevBench/price_vs_consistency")

    plot_judge_level_price_vs_positional_consistency(results_df=DevBench_results_df,
                                                    benchmark='DevBench',
                                                    graph_title='DevBench Price vs. Positional Consistency',
                                                    input_pricing_map=input_pricing_map,
                                                    output_pricing_map=output_pricing_map,
                                                    save_to_directory="DevBench/price_vs_consistency")