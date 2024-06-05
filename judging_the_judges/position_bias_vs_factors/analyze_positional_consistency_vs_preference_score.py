'''
The function analyzes relationship between positional consistency and preference scores with a linear regression
for each judge and overall, and save two graphs - one for Overall and another aggregated graph (3x3) for the nine judges.

The judges are sorted based on the absolute slope of the linear regression line in ascending order.

The `benchmark` parameter specifies whether the analysis is for MTBench or DevBench.

The resulting figures are saved to the specified `save_to_directory` if provided, otherwise they are displayed.

Note: These functions are designed for plotting the results of 9 judges with a specific order for better layout. Feel free to revise to fit specific needs.
'''

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


def analyze_positional_consistency_vs_preference_score_linear(results_df, benchmark, graph_title, save_to_directory=None):
    """
    Analyzes the linear relationship between positional consistency and preference scores
    for each judge and overall, and saves two graphs - one for Overall and another aggregated graph (3x3) for the nine judges.

    Parameters:
    - results_df (pd.DataFrame): The results DataFrame containing columns: Judge, Model, Task, Positional Consistency, Primacy Count, Recency Count.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the file names.
    - graph_title (str): The title of the aggregated graph.
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

    slopes = []

    for judge in top_judges + mid_judges + bottom_judges:
        judge_df = results_df[results_df['Judge'] == judge].copy()  # Create a copy of the filtered DataFrame

        if not judge_df.empty:
            X = judge_df[['Positional Consistency']]
            y = judge_df['Positional Preference Score']
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            slopes.append((judge, abs(slope)))
        else:
            slopes.append((judge, 0))

    # Sort judges based on the absolute slope in ascending order
    sorted_judges = [x[0] for x in sorted(slopes, key=lambda x: x[1])]

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.suptitle(f"{graph_title}", fontsize=16)

    for i, judge in enumerate(sorted_judges):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        judge_df = results_df[results_df['Judge'] == judge].copy()  # Create a copy of the filtered DataFrame

        # Count the occurrences of each (x, y) pair
        xy_counts = judge_df.groupby(['Positional Consistency', 'Positional Preference Score']).size().reset_index(name='count')

        # Scatter plot with color intensity based on count
        sc = ax.scatter(xy_counts['Positional Consistency'], xy_counts['Positional Preference Score'], c=xy_counts['count'], cmap='viridis')
        ax.set_title(f'{judge}')
        ax.set_xlabel('Positional Consistency')
        ax.set_ylabel('Positional Preference Score')
        
        # Draw the horizontal baseline at Positional Preference Score = 0
        ax.axhline(y=0, color='black', linestyle='--')

        # Linear Regression
        if not judge_df.empty:
            X = judge_df[['Positional Consistency']]
            y = judge_df['Positional Preference Score']
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            # Sort values for plotting
            sorted_zip = sorted(zip(X['Positional Consistency'], y_pred))
            X_plot, y_pred_plot = zip(*sorted_zip)
            ax.plot(X_plot, y_pred_plot, color='red')
            score = model.score(X, y)
            ax.text(0.75, 0.95, f'R^2: {score:.2f}', transform=ax.transAxes, fontsize=9, verticalalignment='top')

    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.set_ylabel('Count', rotation=90)

    if save_to_directory:
        fig_path = os.path.join(save_to_directory, f"{benchmark}_positional_consistency_vs_preference_score_judges.png")
        plt.savefig(fig_path)
    else:
        plt.show()

    # Plot Overall graph
    overall_df = results_df.copy()  # Create a copy of the entire DataFrame
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count the occurrences of each (x, y) pair
    xy_counts = overall_df.groupby(['Positional Consistency', 'Positional Preference Score']).size().reset_index(name='count')

    # Scatter plot with color intensity based on count
    sc = ax.scatter(xy_counts['Positional Consistency'], xy_counts['Positional Preference Score'], c=xy_counts['count'], cmap='viridis')
    ax.set_title(f'{graph_title} (Overall)')
    ax.set_xlabel('Positional Consistency')
    ax.set_ylabel('Positional Preference Score')
    
    # Draw the horizontal baseline at Positional Preference Score = 0
    ax.axhline(y=0, color='black', linestyle='--')
    
    # Add colorbar
    cbar = fig.colorbar(sc)
    cbar.set_label('Count')

    # Linear Regression
    if not overall_df.empty:
        X = overall_df[['Positional Consistency']]
        y = overall_df['Positional Preference Score']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        # Sort values for plotting
        sorted_zip = sorted(zip(X['Positional Consistency'], y_pred))
        X_plot, y_pred_plot = zip(*sorted_zip)
        ax.plot(X_plot, y_pred_plot, color='red')
        score = model.score(X, y)
        ax.text(0.75, 0.95, f'R^2: {score:.2f}', transform=ax.transAxes, fontsize=9, verticalalignment='top')

        # Add 'primacy' and 'recency' labels to the y-axis
        ax.set_ylim(-1.3, 1.3)
        ylim = ax.get_ylim()
        ax.text(0.1, ylim[0]+0.1, 'primacy ↓', ha='left', va='bottom', fontsize=12)
        ax.text(0.1, ylim[1]-0.1, 'recency ↑', ha='left', va='top', fontsize=12)

    if save_to_directory:
        fig_path = os.path.join(save_to_directory, f"{benchmark}_positional_consistency_vs_preference_score_overall.png")
        plt.savefig(fig_path)
    else:
        plt.tight_layout()
        plt.show()

    print(f"{benchmark} analyze_positional_consistency_vs_preference_score_linear complete.")
    
if __name__=="__main__":
    MTBench_results_with_length_stats = pd.read_csv('MTBench/(Judge-Model-Task)_results.csv')
    analyze_positional_consistency_vs_preference_score_linear(results_df=MTBench_results_with_length_stats, 
                                                              benchmark='MTBench',
                                                              graph_title='MTBench Positional Consistency vs. Positional Preference Score',
                                                              save_to_directory="MTBench/consistency_vs_preference")
    
    DevBench_results_with_length_stats = pd.read_csv('DevBench/(Judge-Model-Task)_results.csv')
    analyze_positional_consistency_vs_preference_score_linear(results_df=DevBench_results_with_length_stats,
                                                              benchmark='DevBench',
                                                              graph_title='DevBench Positional Consistency vs. Positional Preference Score',
                                                              save_to_directory="DevBench/consistency_vs_preference")