'''
These functions analyze positional consistency and preference scores vs. length stats in (Judge, Model, Task) unit. 
The analysis plots the metric vs. logarithm of lengths as y and x axis respectively using heated scatter plot.

`nonlinear` refers to a polynomial nonlinear regression of the data points. 
`benchmark` does not make any difference in the analysis part - it's just a part of graph title and file names.

Run these functions after calculating the corresponding length stats.

Usage:
- If you have augmented all length stats (input length, output length, prompt length), then you may call the uppermost integrated function.
- If you only want to run one of them, then you may call the corresponding ones beneath.
- These functions are designed for exactly 9 judges with a 3x3 subfigure layout. Feel free to revise them to your specific needs.
- You may also create and save the separate analysis for each judge on your own. Here for convenience, we just provide the integrated version.

IMPORTANT:
- Lengths are considered Task-level factors that influence position bias
- We use logarithm of lengths to better visualize the relationship and to reduce the impact of extreme values
- Expectation:
  - We observe no concluding patterns.
  - We also tried analysis on each Judge-Task level, or using a linear regression, but still no concluding patterns
'''

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def analyze_positional_consistency_and_preference_score_vs_log_length_stat_nonlinear(results_df, benchmark, save_to_directory=None):
    analyze_positional_consistency_and_preference_score_vs_log_input_length_nonlinear(results_df, benchmark, f"{save_to_directory}/log_input_length")
    analyze_positional_consistency_and_preference_score_vs_log_output_length_nonlinear(results_df, benchmark, f"{save_to_directory}/log_output_length")
    analyze_positional_consistency_and_preference_score_vs_log_prompt_length_nonlinear(results_df, benchmark, f"{save_to_directory}/log_prompt_length")


def analyze_positional_consistency_and_preference_score_vs_log_input_length_nonlinear(results_df, benchmark, save_to_directory=None):
    """
    Analyzes the relationship between positional consistency and preference scores over the logarithm of input length for each judge.
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns:
      Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count, avg_task_input_length.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the graph title.
    - save_to_directory (str, optional): The directory path to save the generated figures. If not provided, the figures will be displayed.
      
    Returns:
    - None
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    top_judges = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613']
    mid_judges = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gemini-pro']
    bottom_judges = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    
    for score_type in ['Positional Consistency', 'Positional Preference Score']:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
        fig.suptitle(f"{benchmark} \n {score_type} vs. Log Input Length", fontsize=16)
            
        for i, judge in enumerate(top_judges + mid_judges + bottom_judges):
            judge_df = results_df[results_df['Judge'] == judge].copy()
            
            row, col = i // 3, i % 3
            
            # Take the logarithm of input length
            judge_df['log_avg_task_input_length'] = np.log(judge_df['avg_task_input_length'])
            
            # Count the occurrences of each (x, y) pair
            xy_counts = judge_df.groupby(['log_avg_task_input_length', score_type]).size().reset_index(name='count')
            
            # Scatter plot with color intensity based on count
            sc = axs[row, col].scatter(xy_counts['log_avg_task_input_length'], xy_counts[score_type], c=xy_counts['count'], cmap='viridis')
            axs[row, col].set_title(f'{judge}')
            axs[row, col].set_xlabel('Log Average Task Input Length')
            axs[row, col].set_ylabel(score_type)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=axs[row, col])
            cbar.set_label('Count')
            
            # Polynomial Regression
            if not judge_df.empty:
                X = judge_df[['log_avg_task_input_length']]
                y = judge_df[score_type]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Sort values for plotting
                sorted_zip = sorted(zip(X['log_avg_task_input_length'], y_pred))
                X_plot, y_pred_plot = zip(*sorted_zip)
                
                axs[row, col].plot(X_plot, y_pred_plot, color='red')
                score = r2_score(y, y_pred)
                axs[row, col].text(0.05, 0.95, f'R^2: {score:.2f}', transform=axs[row, col].transAxes, 
                                   fontsize=9, verticalalignment='top')
                
        plt.tight_layout()
        if save_to_directory:
            fig_name = f"{benchmark}_{score_type}_vs_log_input_length_all.png"
            fig_path = os.path.join(save_to_directory, fig_name)
            plt.savefig(fig_path)
        else:        
            plt.show()
        # Close the figure after saving or displaying it
        plt.close(fig)
    print(f"{benchmark} analyze_positional_consistency_and_preference_score_vs_log_input_length_nonlinear complete.")


def analyze_positional_consistency_and_preference_score_vs_log_output_length_nonlinear(results_df, benchmark, save_to_directory=None):
    """
    Analyzes the relationship between positional consistency and preference scores over the logarithm of output length for each judge.
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns:
      Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count, avg_task_output_length.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the graph title.
    - save_to_directory (str, optional): The directory path to save the generated figures. If not provided, the figures will be displayed.
      
    Returns:
    - None
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    top_judges = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613']
    mid_judges = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gemini-pro']
    bottom_judges = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    
    for score_type in ['Positional Consistency', 'Positional Preference Score']:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
        fig.suptitle(f"{benchmark} \n {score_type} vs. Log Output Length", fontsize=16)
            
        for i, judge in enumerate(top_judges + mid_judges + bottom_judges):
            judge_df = results_df[results_df['Judge'] == judge].copy()
            
            row, col = i // 3, i % 3
            
            # Take the logarithm of output length
            judge_df['log_avg_task_output_length'] = np.log(judge_df['avg_task_output_length'])
            
            # Count the occurrences of each (x, y) pair
            xy_counts = judge_df.groupby(['log_avg_task_output_length', score_type]).size().reset_index(name='count')
            
            # Scatter plot with color intensity based on count
            sc = axs[row, col].scatter(xy_counts['log_avg_task_output_length'], xy_counts[score_type], c=xy_counts['count'], cmap='viridis')
            axs[row, col].set_title(f'{judge}')
            axs[row, col].set_xlabel('Log Average Task Output Length')
            axs[row, col].set_ylabel(score_type)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=axs[row, col])
            cbar.set_label('Count')
            
            # Polynomial Regression
            if not judge_df.empty:
                X = judge_df[['log_avg_task_output_length']]
                y = judge_df[score_type]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Sort values for plotting
                sorted_zip = sorted(zip(X['log_avg_task_output_length'], y_pred))
                X_plot, y_pred_plot = zip(*sorted_zip)
                
                axs[row, col].plot(X_plot, y_pred_plot, color='red')
                score = r2_score(y, y_pred)
                axs[row, col].text(0.05, 0.95, f'R^2: {score:.2f}', transform=axs[row, col].transAxes, 
                                   fontsize=9, verticalalignment='top')
                
        plt.tight_layout()
        if save_to_directory:
            fig_name = f"{benchmark}_{score_type}_vs_log_output_length_all.png"
            fig_path = os.path.join(save_to_directory, fig_name)
            plt.savefig(fig_path)
        else:        
            plt.show()
        # Close the figure after saving or displaying it
        plt.close(fig)
    print(f"{benchmark} analyze_positional_consistency_and_preference_score_vs_log_output_length_nonlinear complete.")


def analyze_positional_consistency_and_preference_score_vs_log_prompt_length_nonlinear(results_df, benchmark, save_to_directory=None):
    """
    Analyzes the relationship between positional consistency and preference scores over the logarithm of prompt length for each judge.
    
    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns:
      Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count, avg_prompt_length.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the graph title.
    - save_to_directory (str, optional): The directory path to save the generated figures. If not provided, the figures will be displayed.
      
    Returns:
    - None
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    top_judges = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613']
    mid_judges = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gemini-pro']
    bottom_judges = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    
    for score_type in ['Positional Consistency', 'Positional Preference Score']:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
        fig.suptitle(f"{benchmark} \n {score_type} vs. Log Prompt Length", fontsize=16)
            
        for i, judge in enumerate(top_judges + mid_judges + bottom_judges):
            judge_df = results_df[results_df['Judge'] == judge].copy()
            
            row, col = i // 3, i % 3
            
            # Take the logarithm of prompt length
            judge_df['log_avg_prompt_length'] = np.log(judge_df['avg_prompt_length'])
            
            # Count the occurrences of each (x, y) pair
            xy_counts = judge_df.groupby(['log_avg_prompt_length', score_type]).size().reset_index(name='count')
            
            # Scatter plot with color intensity based on count
            sc = axs[row, col].scatter(xy_counts['log_avg_prompt_length'], xy_counts[score_type], c=xy_counts['count'], cmap='viridis')
            axs[row, col].set_title(f'{judge}')
            axs[row, col].set_xlabel('Log Average Prompt Length')
            axs[row, col].set_ylabel(score_type)
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=axs[row, col])
            cbar.set_label('Count')
            
            # Polynomial Regression
            if not judge_df.empty:
                X = judge_df[['log_avg_prompt_length']]
                y = judge_df[score_type]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Sort values for plotting
                sorted_zip = sorted(zip(X['log_avg_prompt_length'], y_pred))
                X_plot, y_pred_plot = zip(*sorted_zip)
                
                axs[row, col].plot(X_plot, y_pred_plot, color='red')
                score = r2_score(y, y_pred)
                axs[row, col].text(0.05, 0.95, f'R^2: {score:.2f}', transform=axs[row, col].transAxes, 
                                   fontsize=9, verticalalignment='top')
                
        plt.tight_layout()
        if save_to_directory:
            fig_name = f"{benchmark}_{score_type}_vs_log_prompt_length_all.png"
            fig_path = os.path.join(save_to_directory, fig_name)
            plt.savefig(fig_path)
        else:        
            plt.show()
        # Close the figure after saving or displaying it
        plt.close(fig)
    print(f"{benchmark} analyze_positional_consistency_and_preference_score_vs_log_prompt_length_nonlinear complete.")


if __name__=="__main__":
    MTBench_results_with_length_stats = pd.read_csv('MTBench/results_with_both_winrates_and_length_stats.csv')
    analyze_positional_consistency_and_preference_score_vs_log_length_stat_nonlinear(results_df=MTBench_results_with_length_stats, 
                                                                                      benchmark='MTBench',
                                                                                      save_to_directory='MTBench/lengths')
    
    DevBench_results_with_length_stats = pd.read_csv('DevBench/results_with_both_winrates_and_length_stats.csv')
    analyze_positional_consistency_and_preference_score_vs_log_length_stat_nonlinear(results_df=DevBench_results_with_length_stats,
                                                                                      benchmark='DevBench',
                                                                                      save_to_directory='DevBench/lengths')