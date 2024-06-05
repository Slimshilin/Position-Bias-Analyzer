'''
These functions analyze positional consistency and preference scores vs. win rates in (Judge, Model, Task) unit.
The analysis plots the metric vs. win rates as y and x axis respectively using heated scatter plot.
The baselines (i.e., 0.5 win rate and 0 positional preference score) are drawn in ------
`nonlinear` referes to a polynomial nonlinear regression of the data points.
`benchmark` does not make any difference in the analysis part - it's just a part of graph title and file names.

Run these functions after calculating the corresponding win rates.

Usage:
    - If you have augmented both win rates, then you may call the uppermost integrated function.
    - If you only only want to run one of them, then you may call the corresponding ones beneath.
    - These functions are designed for exactly 9 judges with a 3x3 subfigure layout.
        Feel free to revise them to your specific needs. You may also create and save the seperate analysis for each judge on your own.
        Here for convenience, we just provide the integrated version.

IMPORTANT:
    - Win rates are considered Model-level factors that influence position bias
    - We use overall win rate to measure the answer quality gap (i.e., the quality disparity of answers between the Model and the reference model)
    - Answer quality gap could be measured by |overall win rate - 0.5|. 
        However, since it's very convenient and it provides more information, we directly use overall win rate without furhter calculating quality gap.
    - Expectation:
        - Positonal Consistency vs. overall win rate should give a parabolic shape regression, 
            implying higher answer quality gap generally results in higher positional consistency
        - Positional Preference Score vs. overall may not have clear pattern
        - consistent win rate will not exhibit as clear pattern as overall win rate
'''


import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def analyze_positional_consistency_and_preference_score_vs_winrates_nonlinear(results_df, benchmark, save_to_directory=None):
    analyze_positional_consistency_and_preference_score_vs_consistent_winrate_nonlinear(results_df, benchmark, f"{save_to_directory}/consistent_winrate")
    analyze_positional_consistency_and_preference_score_vs_overall_winrate_nonlinear(results_df, benchmark, f"{save_to_directory}/overall_winrate")

def analyze_positional_consistency_and_preference_score_vs_consistent_winrate_nonlinear(results_df, benchmark, save_to_directory=None):
    """
    Analyzes the relationship between positional consistency/preference scores and consistent win rates using scatter plots with color intensity based on count.
    Plots separate subplots for each judge in a 3x3 grid layout.
    
    Note: This function is designed for plotting the results of 9 judges with a specific order for better layout. Feel free to revise to fit specific needs.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the results with columns: 'Judge', 'Positional Consistency', 'Positional Preference Score', 'consistent_winrate'.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the graph title.
    - save_to_directory (str, optional): Directory to save the plots. If None, plots will be displayed instead of saved.
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    top_judges = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613']
    mid_judges = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gemini-pro']
    bottom_judges = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    
    for score_type in ['Positional Consistency', 'Positional Preference Score']:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
        fig.suptitle(f"{benchmark} \n {score_type} vs. Consistent Win Rate", fontsize=16)
        
        for i, judge in enumerate(top_judges + mid_judges + bottom_judges):
            judge_df = results_df[results_df['Judge'] == judge]
            
            row, col = i // 3, i % 3
            
            # Count the occurrences of each (x, y) pair
            xy_counts = judge_df.groupby(['consistent_winrate', score_type]).size().reset_index(name='count')
            
            # Scatter plot with color intensity based on count
            sc = axs[row, col].scatter(xy_counts['consistent_winrate'], xy_counts[score_type], c=xy_counts['count'], cmap='viridis')
            axs[row, col].set_title(f'{judge}')
            axs[row, col].set_xlabel('Consistent Win Rate')
            axs[row, col].set_ylabel(score_type)
            
            # Add vertical line at 0.5
            axs[row, col].axvline(x=0.5, color='black', linestyle='--')

            if score_type == 'Positional Preference Score':
                axs[row, col].axhline(y=0, color='black', linestyle='-')
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=axs[row, col])
            cbar.set_label('Count')
            
            # Polynomial Regression
            if not judge_df.empty:
                X = judge_df[['consistent_winrate']]
                y = judge_df[score_type]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Sort values for plotting
                sorted_zip = sorted(zip(X['consistent_winrate'], y_pred))
                X_plot, y_pred_plot = zip(*sorted_zip)
                
                axs[row, col].plot(X_plot, y_pred_plot, color='red')
                score = r2_score(y, y_pred)
                axs[row, col].text(0.05, 0.95, f'R^2: {score:.2f}', transform=axs[row, col].transAxes, 
                                   fontsize=9, verticalalignment='top')
            
        plt.tight_layout()
        
        if save_to_directory:
            fig_name = f"{benchmark}_{score_type}_vs_consistent_winrate_all.png"
            fig_path = os.path.join(save_to_directory, fig_name)
            plt.savefig(fig_path)
        else:        
            plt.show()
            
    print(f"{benchmark} analyze_positional_consistency_and_preference_score_vs_consistent_winrate_nonlinear complete.")

def analyze_positional_consistency_and_preference_score_vs_overall_winrate_nonlinear(results_df, benchmark, save_to_directory=None):
    """
    Analyzes the relationship between positional consistency/preference scores and overall win rates using scatter plots with color intensity based on count.
    Plots separate subplots for each judge in a 3x3 grid layout.
    
    Note: This function is designed for plotting the results of 9 judges with a specific order for better layout. Feel free to revise to fit specific needs.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the results with columns: 'Judge', 'Positional Consistency', 'Positional Preference Score', 'overall_winrate'.
    - benchmark (str): The name of the benchmark (e.g., 'MTBench', 'DevBench') used for the graph title.
    - save_to_directory (str, optional): Directory to save the plots. If None, plots will be displayed instead of saved.
    """
    # create dir if not exist
    if save_to_directory and not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)
        
    top_judges = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613']
    mid_judges = ['gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 'gemini-pro']
    bottom_judges = ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
    
    for score_type in ['Positional Consistency', 'Positional Preference Score']:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True)
        fig.suptitle(f"{benchmark} \n {score_type} vs. Overall Win Rate", fontsize=16)
        
        for i, judge in enumerate(top_judges + mid_judges + bottom_judges):
            judge_df = results_df[results_df['Judge'] == judge]
            
            row, col = i // 3, i % 3
            
            # Count the occurrences of each (x, y) pair
            xy_counts = judge_df.groupby(['overall_winrate', score_type]).size().reset_index(name='count')
            
            # Scatter plot with color intensity based on count
            sc = axs[row, col].scatter(xy_counts['overall_winrate'], xy_counts[score_type], c=xy_counts['count'], cmap='viridis')
            axs[row, col].set_title(f'{judge}')
            axs[row, col].set_xlabel('Overall Win Rate')
            axs[row, col].set_ylabel(score_type)
            
            # Add vertical line at 0.5
            axs[row, col].axvline(x=0.5, color='black', linestyle='--')
            
            # Add horizontal line at 0 for Preference Score
            if score_type == 'Positional Preference Score':
                axs[row, col].axhline(y=0, color='black', linestyle='-')
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=axs[row, col])
            cbar.set_label('Count')
            
            # Polynomial Regression
            if not judge_df.empty:
                X = judge_df[['overall_winrate']]
                y = judge_df[score_type]
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                # Sort values for plotting
                sorted_zip = sorted(zip(X['overall_winrate'], y_pred))
                X_plot, y_pred_plot = zip(*sorted_zip)
                
                axs[row, col].plot(X_plot, y_pred_plot, color='red')
                score = r2_score(y, y_pred)
                axs[row, col].text(0.05, 0.95, f'R^2: {score:.2f}', transform=axs[row, col].transAxes, 
                                   fontsize=9, verticalalignment='top')
            
        plt.tight_layout()
        
        if save_to_directory:
            fig_name = f"{benchmark}_{score_type}_vs_overall_winrate_all.png"
            fig_path = os.path.join(save_to_directory, fig_name)
            plt.savefig(fig_path)
        else:        
            plt.show()
            
    print(f"{benchmark} analyze_positional_consistency_and_preference_score_vs_overall_winrate_nonlinear complete.")

if __name__=="__main__":
    MTBench_results_with_both_winrates = pd.read_csv('MTBench/results_with_both_winrates.csv')
    analyze_positional_consistency_and_preference_score_vs_winrates_nonlinear(results_df=MTBench_results_with_both_winrates, 
                                                                              benchmark='MTBench',
                                                                              save_to_directory='MTBench/winrate')
    
    DevBench_results_with_both_winrates = pd.read_csv('DevBench/results_with_both_winrates.csv')
    analyze_positional_consistency_and_preference_score_vs_winrates_nonlinear(results_df=DevBench_results_with_both_winrates,
                                                                              benchmark='DevBench',
                                                                              save_to_directory='DevBench/winrate')