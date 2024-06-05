'''
These functions compare the positional consistency and preference scores of a baseline across different tasks.
For positional consistency, it compares other judges' performances with one baseline judge.
For positional preference score, it compares all judges' performances with 0.

The baseline comparison graphs are validated using t-tests with * marking significance

The `benchmark` parameter specifies whether the comparison is for MTBench or DevBench. They are essentially different because of graph parameter settings (font, etc.)
The `baseline_judge` parameter specifies the judge to be used as the baseline for comparison. In both benchmarks, GPT-4 is chosen to be the baseline.

The resulting graphs are saved to the specified `save_to_directory` if provided, otherwise they are displayed.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_baseline_positional_consistency_comparison_graph(results_df, benchmark, graph_title, baseline_judge, save_to_directory=None, x_label_rotation=45, alpha=0.05):
    """
    Plots a comparison graph between the baseline judge and other judges for positional consistency.
    The y-value indicates the percentage difference of each judge model compared to the baseline model
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns:
      Judge, Model, Task, Positional Consistency.
    - benchmark (str): The benchmark name, either "MTBench" or "DevBench".
    - graph_title (str): The title of the graph.
    - baseline_judge (str): The name of the baseline judge to compare against.
    - save_to_directory (str, optional): The directory path to save the generated figure. If not provided, the figure will be displayed.
    - x_label_rotation (int, optional): The rotation angle for the x-axis labels. Default is 45 degrees.
    - alpha (float, optional): The significance level for the t-test. Default is 0.05.
      
    Returns:
    - None
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
    
    # Get the unique judges excluding the baseline judge
    judges = results_df[results_df['Judge'] != baseline_judge]['Judge'].unique()
    
    # Get the unique tasks
    tasks = results_df['Task'].unique()
    
    # Calculate the average positional consistency for each (Judge, Task) combination
    positional_consistency = results_df.groupby(['Judge', 'Task'])['Positional Consistency'].mean().reset_index()
    
    # Get the baseline positional consistency for each task
    baseline_scores = positional_consistency[positional_consistency['Judge'] == baseline_judge].set_index('Task')['Positional Consistency']
    
    # Calculate the percentage difference from the baseline for each (Judge, Task) combination
    positional_consistency['Percentage Difference'] = positional_consistency.apply(lambda row: (row['Positional Consistency'] - baseline_scores[row['Task']]) / baseline_scores[row['Task']] * 100, axis=1)
    
    # Create the plot
    if benchmark == "MTBench":
        fig, ax = plt.subplots(figsize=(16, 10))
    elif benchmark == "DevBench":
        fig, ax = plt.subplots(figsize=(19, 10))
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
    
    # Set the width of each judge's region
    width = 0.8 / len(tasks)
    
    # Define a color palette suitable for academic publications (e.g., NeurIPS)
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot the percentage difference for each (Judge, Task) combination as bars
    for i, task in enumerate(tasks):
        task_data = positional_consistency[positional_consistency['Task'] == task]
        x = np.arange(len(judges))
        bars = ax.bar(x + i * width, task_data[task_data['Judge'].isin(judges)]['Percentage Difference'], width, label=task, color=color_palette[i % len(color_palette)])
        
        # Perform t-test for each bar and add significance markers
        for j, bar in enumerate(bars):
            judge = judges[j]
            judge_task_data = results_df[(results_df['Judge'] == judge) & (results_df['Task'] == task)]['Positional Consistency']
            baseline_task_data = results_df[(results_df['Judge'] == baseline_judge) & (results_df['Task'] == task)]['Positional Consistency']
            _, p_value = stats.ttest_ind(judge_task_data, baseline_task_data)
            
            if p_value < alpha:
                marker_pos = bar.get_height() + 0.5 if bar.get_height() > 0 else bar.get_height() - 1.5
                ax.text(bar.get_x() + bar.get_width() / 2, marker_pos, '*', ha='center', va='bottom' if bar.get_height() > 0 else 'top', fontsize=12)
    
    # Plot the baseline as a horizontal line at y = 0
    ax.axhline(y=0, color='black', linestyle='-', label=baseline_judge, linewidth=2.25)
    
    # Set the plot title and labels
    ax.set_title(graph_title, fontsize=18)
    ax.set_xlabel("Judge", fontsize=16)
    ax.set_ylabel(f"Percentage Difference from {baseline_judge}", fontsize=16)
    
    # Set the x-tick labels to the judge names and rotate them
    ax.set_xticks(x + (len(tasks) - 1) * width / 2)
    ax.set_xticklabels(judges, rotation=x_label_rotation, ha='right', fontsize=16)
    
    if benchmark == "MTBench":
        ax.legend(title='Task', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=20)
    elif benchmark == "DevBench":
        ax.legend(title='Task', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, title_fontsize=18)
    
    # Set the y-axis tick label size
    ax.tick_params(axis='y', labelsize=12)
    
    if benchmark == "MTBench":
        ax.set_ylim(-60, 10)  # Manualy set the y-axis limits to range from -60 to 10 for better visualization
    
    # Adjust the plot layout
    plt.tight_layout()
    
    if save_to_directory:
        fig_name = f"{benchmark}_baseline_positional_consistency_comparison.png"
        fig_path = os.path.join(save_to_directory, fig_name)
        plt.savefig(fig_path, dpi=300)
    else:
        plt.show()
    
    print(f"{benchmark} positional consistency baseline comparison graph saved to {fig_path}")


def plot_baseline_positional_preference_score_comparison_graph(results_df, benchmark, graph_title, save_to_directory=None, x_label_rotation=45, alpha=0.05):
    """
    Plots a comparison graph of all judges compared to the 0 positonal preference score baseline.
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns:
      Judge, Model, Task, Positional Preference Score.
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The title of the graph.
    - save_to_directory (str, optional): The directory path to save the generated figure. If not provided, the figure will be displayed.
    - x_label_rotation (int, optional): The rotation angle for the x-axis labels. Default is 45 degrees.
    - alpha (float, optional): The significance level for the t-test. Default is 0.05.
      
    Returns:
    - None
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
    
    # Get the unique judges
    judges = results_df['Judge'].unique()
    
    # Get the unique tasks
    tasks = results_df['Task'].unique()
    
    # Filter out consistency=1 data
    results_df = results_df[results_df['Positional Consistency'] != 1]
    
    # Calculate the average positional preference score for each (Judge, Task) combination
    preference_scores = results_df.groupby(['Judge', 'Task'])['Positional Preference Score'].mean().reset_index()
    
    # Create the plot
    if benchmark == "MTBench":
        fig, ax = plt.subplots(figsize=(16, 10))
    elif benchmark == "DevBench":
        fig, ax = plt.subplots(figsize=(19, 10))
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
    
    # Set the width of each judge's region
    width = 0.8 / len(tasks)
    
    # Define a color palette suitable for academic publications (e.g., NeurIPS)
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot the preference scores for each (Judge, Task) combination as bars
    for i, task in enumerate(tasks):
        task_data = preference_scores[preference_scores['Task'] == task]
        x = np.arange(len(judges))
        bars = ax.bar(x + i * width, task_data[task_data['Judge'].isin(judges)]['Positional Preference Score'], width, label=task, color=color_palette[i % len(color_palette)])
        
        # Perform t-test for each bar and add significance markers
        for j, bar in enumerate(bars):
            judge = judges[j]
            judge_task_data = results_df[(results_df['Judge'] == judge) & (results_df['Task'] == task)]['Positional Preference Score']
            _, p_value = stats.ttest_1samp(judge_task_data, 0)
            
            if p_value < alpha:
                if bar.get_height() > 0:
                    # Place the * marker at the top of the bar
                    marker_pos = bar.get_height() + 0.02
                    va_align = 'bottom'
                else:
                    # For below baseline bars, place the * marker at the bottom of the bar
                    marker_pos = bar.get_height() - 0.02
                    va_align = 'top'

                # Set the position of the text inside the bar
                if benchmark == "MTBench":
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            marker_pos, '*', 
                            ha='center', 
                            va=va_align, 
                            fontsize=12)
                elif benchmark == "DevBench":
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            marker_pos, '*', 
                            ha='center', 
                            va=va_align, 
                            fontsize=10)                    
                        
    # Plot the baseline of 0 as a horizontal line
    ax.axhline(y=0, color='black', linestyle='-', label='Baseline', linewidth=2.25)
    
    # Set the plot title and labels
    ax.set_title(graph_title, fontsize=18)
    ax.set_xlabel("Judge", fontsize=16)
    ax.set_ylabel("Positional Preference Score", fontsize=16)
    
    # Set the x-tick labels to the judge names and rotate them
    ax.set_xticks(x + (len(tasks) - 1) * width / 2)
    ax.set_xticklabels(judges, rotation=x_label_rotation, ha='right', fontsize=16)
    
    ylim = ax.get_ylim()
    if benchmark == "MTBench":
        ax.legend(title='Task', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=16, title_fontsize=20)
    elif benchmark == "DevBench":
        ax.legend(title='Task', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, title_fontsize=18)
        # Manually to set the limit for first for prettier layout
        ax.set_ylim(-0.4,1.1)
        ylim = ax.get_ylim()

    # Add 'primacy' and 'recency' labels to the y-axis
    ax.text(0, ylim[0], 'primacy ↓', ha='left', va='bottom', fontsize=17)
    ax.text(0, ylim[1], 'recency ↑', ha='left', va='top', fontsize=17)
    
    # Set the y-axis tick label size
    ax.tick_params(axis='y', labelsize=12)
    
    # Adjust the y-axis limits to create some whitespace at the top and bottom
    ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.1)

    # Adjust the plot layout
    plt.tight_layout()
    
    if save_to_directory:
        fig_name = f"{benchmark}_baseline_positional_preference_score_comparison.png"
        fig_path = os.path.join(save_to_directory, fig_name)
        plt.savefig(fig_path, dpi=300)
    else:
        plt.show()
    
    print(f"{benchmark} positional preference score baseline comparison graph saved to {fig_path}")
    
if __name__=="__main__":
    MTBench_results_df = pd.read_csv('MTBench/(Judge-Model-Task)_results.csv')
    DevBench_results_df = pd.read_csv('DevBench/(Judge-Model-Task)_results.csv')
    
    plot_baseline_positional_consistency_comparison_graph(results_df=MTBench_results_df, 
                                                         benchmark='MTBench',
                                                         graph_title='MTBench Positional Consistency Baseline Comparison',
                                                         baseline_judge='gpt-4-0613',
                                                         save_to_directory='MTBench/baseline-comparison')
    plot_baseline_positional_preference_score_comparison_graph(results_df=MTBench_results_df,
                                                              benchmark='MTBench', 
                                                              graph_title='MTBench Positional Preference Score Baseline Comparison',
                                                              save_to_directory='MTBench/baseline-comparison')
    
    plot_baseline_positional_consistency_comparison_graph(results_df=DevBench_results_df, 
                                                         benchmark='DevBench',
                                                         graph_title='DevBench Positional Consistency Baseline Comparison',
                                                         baseline_judge='gpt-4-0613',
                                                         save_to_directory='DevBench/baseline-comparison')
    plot_baseline_positional_preference_score_comparison_graph(results_df=DevBench_results_df,
                                                              benchmark='DevBench',
                                                              graph_title='DevBench Positional Preference Score Baseline Comparison',
                                                              save_to_directory='DevBench/baseline-comparison')