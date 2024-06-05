'''
These functions conduct Maximum Likelihood Evaluation (MLE) analysis for positional preference scores.
It supports MLE for
1. each Judge
2. each Judge on each Task

These functions should run after calculating the positional consistency and preference scores.

save_to_directory, when specified, would save the graphs to it (without showing); otherwise, it'll show the graphs.

Since the graph parameters (font size, label names, etc.) vary across datasets, it's recommended to adjust them according to your need.
Particularly, for those of judges that are extremely biased on one side of preference, it's better to truncate the x-axis to 99 or 98 instead of 100.
Also, the subfigure organization should be specified according to the number of judges evaluated on each dataset.
Here we provide `benchmark` as the parameter to fit our parameter settings for MTBench and DevBench correspondingly.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.interpolate import interp1d
import math

def binomial_pmf(k, n, p):
    """Binomial probability mass function.
    Args:
        k: Number of successes.
        n: Number of trials.
        p: Probability of success.
    Returns:
        Probability of k successes in n trials.
    """
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k)) * p ** k * (1 - p) ** (n - k)

def MLE_each_task_each_judge(results_df: pd.DataFrame, benchmark, graph_title, order, save_to_directory=None):
    """
    Visualizes the Maximum Likelihood Estimation (MLE) for each task and each judge using binomial distribution.

    Assumption:
    - positional preference scores have been calculated for each (Judge, Model, Task) unit
    - the current function builds on 9 judges with 3x3 subfigure. Revise as needed

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing judges, tasks, and positional preference scores.
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The title of the graph.
    - order (List[str]): The order of judges to be plotted.
    - save_to_directory (str, optional): Directory to save the figure. If None, the figure is displayed. Defaults to None.

    Returns:
    - None
    """
    
    judges = results_df['Judge'].unique()
    tasks = results_df['Task'].unique()
    
    if benchmark == "MTBench":
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(graph_title, fontsize=20)

        colors = plt.colormaps['tab10'](range(len(tasks)))
        
        for i, judge in enumerate(order):
            ax = axes[i // 3, i % 3]
            
            for j, task in enumerate(tasks):
                task_df = results_df[(results_df['Judge'] == judge) & (results_df['Task'] == task)]
                n = len(task_df)
                k = len(task_df[task_df['Positional Preference Score'] > 0])
                
                ps = []
                pmfs = []
                for p in range(0, 100, 1):
                    pmf = binomial_pmf(k, n, p/100.0)
                    ps.append(p)
                    pmfs.append(pmf)
                
                ax.plot(ps, pmfs, color=colors[j], label=task if i == 0 else None)
            
            ax.axvline(x=50, color='black', linestyle='--', label='random' if i == 0 else None)
            ax.set_title(judge, fontsize=15)
            ax.tick_params(axis='both', labelsize=13)
        
        fig.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fontsize=14)
        # plt.tight_layout()
    
    elif benchmark == "DevBench":
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(graph_title, fontsize=20)

        colors = plt.colormaps['tab10'](range(len(tasks)))
        
        for i, judge in enumerate(order):
            ax = axes[i // 3, i % 3]
            
            for j, task in enumerate(tasks):
                task_df = results_df[(results_df['Judge'] == judge) & (results_df['Task'] == task)]
                n = len(task_df)
                k = len(task_df[task_df['Positional Preference Score'] > 0])
                
                ps = []
                pmfs = []
                for p in range(0, 100, 1):
                    pmf = binomial_pmf(k, n, p/100.0)
                    ps.append(p)
                    pmfs.append(pmf)
                
                ax.plot(ps, pmfs, color=colors[j], label=task if i == 0 else None)
            
            ax.axvline(x=50, color='black', linestyle='--', label='random' if i == 0 else None)
            ax.set_title(judge, fontsize=15)
            ax.tick_params(axis='both', labelsize=13)
        
        fig.legend(loc='upper right', bbox_to_anchor=(1.2, 0.9), fontsize=14)
        # plt.tight_layout()
    
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")

    if save_to_directory:
        os.makedirs(save_to_directory, exist_ok=True)
        plt.savefig(f"{save_to_directory}/{benchmark}_MLE_each_task_each_judge.png", bbox_inches='tight')
        print(f"MLE for each task and each judge saved to {save_to_directory}/{benchmark}_MLE_each_task_each_judge.png")
    else:
        plt.show()

def MLE_each_judge(results_df: pd.DataFrame, benchmark, graph_title, save_to_directory=None):
    """
    Visualizes the Maximum Likelihood Estimation (MLE) for each judge using binomial distribution.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing judges and positional preference scores.
    - benchmark (str): The benchmark type, either "MTBench" or "DevBench".
    - graph_title (str): The title of the graph.
    - save_to_directory (str, optional): Directory to save the figure. If None, the figure is displayed. Defaults to None.

    Returns:
    - None
    """
    judges = results_df['Judge'].unique()

    if benchmark == "MTBench":
        min_likelihood_value = 0
        max_likelihood_value = 100
    elif benchmark == "DevBench":
        min_likelihood_value = 0
        max_likelihood_value = 99
    else:
        raise ValueError(f"Input Benchmark {benchmark} not supported. Should be either MTBench or DevBench.")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(graph_title, fontsize=18)
    
    for judge in judges:
        judge_df = results_df[results_df['Judge'] == judge]
        n = len(judge_df)
        k = len(judge_df[judge_df['Positional Preference Score'] > 0])
        
        ps = []
        pmfs = []
        for p in np.linspace(min_likelihood_value, max_likelihood_value, 500):
            pmf = binomial_pmf(k, n, p/100.0)
            ps.append(p)
            pmfs.append(pmf)
        
        f = interp1d(ps, pmfs, kind='cubic')
        ps_smooth = np.linspace(min_likelihood_value, max_likelihood_value, 500)
        pmfs_smooth = f(ps_smooth)
        
        ax.plot(ps_smooth, pmfs_smooth, label=judge)
    
    ax.axvline(x=50, color='black', linestyle='--', label='random')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    # Add 'primacy' and 'recency' labels to the x-axis
    ax.text(0.05, -0.1, 'Primacy', transform=ax.transAxes, fontsize=14, ha='left', va='top')
    ax.text(0.95, -0.1, 'Recency', transform=ax.transAxes, fontsize=14, ha='right', va='top')
    ax.text(0.50, -0.1, 'Fair', transform=ax.transAxes, fontsize=14, ha='center', va='top')
    plt.tight_layout()
    
    if save_to_directory:
        os.makedirs(save_to_directory, exist_ok=True)
        plt.savefig(f"{save_to_directory}/{benchmark}_MLE_each_judge.png", bbox_inches='tight')
        print(f"MLE for each judge saved to {save_to_directory}/{benchmark}_MLE_each_judge.png")
    else:
        plt.show()
    
    
if __name__=="__main__":
    MTBench_results_df = pd.read_csv('MTBench/(Judge-Model-Task)_results.csv')
    DevBench_results_df = pd.read_csv('DevBench/(Judge-Model-Task)_results.csv')
    order = [
        'gpt-4-0125-preview',
        'gpt-4-1106-preview',
        'gpt-4-0613',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'gemini-pro',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
    ]
    MLE_each_task_each_judge(results_df=MTBench_results_df,
                             benchmark='MTBench',
                             graph_title='MTBench MLE for Each Judge and Task',
                             order=order,
                             save_to_directory='MTBench/MLE')
    
    MLE_each_judge(results_df=MTBench_results_df,
                    benchmark='MTBench',
                    graph_title='MTBench MLE for Each Judge',
                    save_to_directory='MTBench/MLE')
    
    MLE_each_task_each_judge(results_df=DevBench_results_df,
                             benchmark='DevBench',
                             graph_title='DevBench MLE for Each Judge and Task',
                             order=order,
                             save_to_directory='DevBench/MLE')
    
    MLE_each_judge(results_df=DevBench_results_df,
                    benchmark='DevBench',
                    graph_title='DevBench MLE for Each Judge',
                    save_to_directory='DevBench/MLE')