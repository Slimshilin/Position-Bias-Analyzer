'''
These functions visualize the positional consistency and preference scores by Judge and Task.
We can either plot by:
1. Judge-Task: directly compare the same judge's performance across tasks
2. Task-Judge: directly compare all judges' performances on the same task

These functions should run after calculating the positional consistency and preference scores.

save_to_directory, when specified, would save the graphs to it (without showing); otherwise, it'll show the graphs.

Since the graph parameters (font size, label names, etc.) vary across datasets, it's recommended to adjust them according to your need.
Here we provide `benchmark` as the parameter to fit our parameter settings for MTBench and DevBench correspondingly.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_positional_consistency_and_preference_score_by_Judge_Task(results_df, benchmark, graph_title, save_to_directory=None):
    """
    Visualizes positional consistency and preference scores from results_df in two different graphs with horizontal bar charts,
    by judge and task.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing judges, models, tasks, positional consistency, and positional preference scores.
    - graph_title (str): The name of the graph for the title.
    - save_to_directory (str, optional): Directory to save the figures to. If None, figures are displayed instead.
    """
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    sns.set(style="whitegrid")
    all_results_df = results_df
    inconsistent_results_df = results_df[results_df['Positional Consistency'] < 1]
    score_types = ['Positional Consistency', 'Positional Preference Score']
    
    for score in score_types:
        target_results_df = all_results_df if score == 'Positional Consistency' else inconsistent_results_df

        # Visualization by Judge and Task
        if benchmark == "MTBench":
            plt.figure(figsize=(18, 14))
            ax = sns.barplot(y='Judge', x=score, hue='Task', data=target_results_df, orient='h', errorbar='sd', capsize=0.1, err_kws={'linewidth': 1.5})
            ax.set_xlabel(score, fontsize=18)
            ax.set_ylabel('Judge', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if score == 'Positional Preference Score':
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(['primacy' if x == ax.get_xticks()[0] else 'recency' if x == ax.get_xticks()[-1] else f'{x:.2f}' for x in ax.get_xticks()], fontsize=16)
            plt.title(f'{graph_title}', fontsize=20)
            plt.legend(title='Task', fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

        
        elif benchmark == "DevBench":
            plt.figure(figsize=(26, 22))
            ax = sns.barplot(y='Judge', x=score, hue='Task', data=target_results_df, orient='h', errorbar='sd', capsize=0.1, err_kws={'linewidth': 1.5})
            ax.set_xlabel(score, fontsize=18)
            ax.set_ylabel('Judge', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if score == 'Normalized Preference Score':
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(['primacy' if x == ax.get_xticks()[0] else 'recency' if x == ax.get_xticks()[-1] else f'{x:.2f}' for x in ax.get_xticks()], fontsize=16)
            plt.title(f'{graph_title}', fontsize=20)
            plt.legend(title='Task', fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
        
        else:
            raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
        
        # Save or dislay figures
        if save_to_directory:
            plt.savefig(f'{save_to_directory}/{benchmark}_{score.replace(" ", "_")}_by_Judge_and_Task.png')
        else:
            plt.show()

    print("plot_positional_consistency_and_preference_score_by_Judge_Task complete.")
    

def plot_positional_consistency_and_preference_score_by_Task_Judge(results_df, benchmark, graph_title, save_to_directory=None):
    """
    Visualizes positional consistency and preference scores from results_df in two different graphs with horizontal bar charts,
    by task and judge.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing judges, models, tasks, positional consistency, and positional preference scores.
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The name of the graph for the title.
    - save_to_directory (str, optional): Directory to save the figures to. If None, figures are displayed instead.
    """
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)

    sns.set(style="whitegrid")
    all_results_df = results_df
    inconsistent_results_df = results_df[results_df['Positional Consistency'] < 1]
    score_types = ['Positional Consistency', 'Positional Preference Score']

    for score in score_types:
        target_results_df = all_results_df if score == 'Positional Consistency' else inconsistent_results_df

        # Visualization by Task and Judge
        if benchmark == "MTBench":
            plt.figure(figsize=(18, 14))
            ax = sns.barplot(y='Task', x=score, hue='Judge', data=target_results_df, orient='h', errorbar='sd', capsize=0.1, err_kws={'linewidth': 1.5})
            ax.set_xlabel(score, fontsize=18)
            ax.set_ylabel('Task', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if score == 'Positional Preference Score':
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(['primacy' if x == ax.get_xticks()[0] else 'recency' if x == ax.get_xticks()[-1] else f'{x:.2f}' for x in ax.get_xticks()], fontsize=16)
            plt.title(f'{graph_title}', fontsize=20)
            plt.legend(title='Judge', fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
        elif benchmark == "DevBench":
            plt.figure(figsize=(26, 22))
            ax = sns.barplot(y='Task', x=score, hue='Judge', data=target_results_df, orient='h', errorbar='sd', capsize=0.1, err_kws={'linewidth': 1.5})
            ax.set_xlabel(score, fontsize=18)
            ax.set_ylabel('Task', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            if score == 'Positional Preference Score':
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(['primacy' if x == ax.get_xticks()[0] else 'recency' if x == ax.get_xticks()[-1] else f'{x:.2f}' for x in ax.get_xticks()], fontsize=16)
            plt.title(f'{graph_title}', fontsize=20)
            plt.legend(title='Judge', fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
        else:
            raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")

        # Save or display figures
        if save_to_directory:
            plt.savefig(f'{save_to_directory}/{benchmark}_{score.replace(" ", "_")}_by_Task_and_Judge.png')
        else:
            plt.show()

    print("plot_positional_consistency_and_preference_score_by_Task_Judge complete.")


if __name__=="__main__":
    MTBench_results_df = pd.read_csv('MTBench/(Judge-Model-Task)_results.csv')
    plot_positional_consistency_and_preference_score_by_Judge_Task(results_df=MTBench_results_df,
                                                                   benchmark='MTBench',
                                                                   graph_title='MTBench Positional Consistency \n By-Judge-Task',
                                                                   save_to_directory='MTBench/By-Judge_and_Task')
    plot_positional_consistency_and_preference_score_by_Task_Judge(results_df=MTBench_results_df,
                                                                   benchmark='MTBench',
                                                                   graph_title='MTBench Positional Preference Score \n By-Task-Judge',
                                                                   save_to_directory='MTBench/By-Judge_and_Task')
    
    DevBench_results_df = pd.read_csv('DevBench/(Judge-Model-Task)_results.csv')
    plot_positional_consistency_and_preference_score_by_Judge_Task(results_df=DevBench_results_df,
                                                                   benchmark='DevBench',
                                                                   graph_title='DevBench Positional Consistency \n By-Judge-Task',
                                                                   save_to_directory='DevBench/By-Judge_and_Task')
    plot_positional_consistency_and_preference_score_by_Task_Judge(results_df=DevBench_results_df,
                                                                   benchmark='DevBench',
                                                                   graph_title='DevBench Positional Preference Score \n By-Task-Judge',
                                                                   save_to_directory='DevBench/By-Judge_and_Task') 