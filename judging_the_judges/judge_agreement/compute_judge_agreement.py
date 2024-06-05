'''
These function calculates and visualizes the judge mutual agreement matrix.

The input data should have choices extracted (i.e., should have columns named as extracted_{judge})

Judge agreement is only calcualted for both successful extraction.

save_to_file, when specified, would save the agreement matrix to it; otherwise, it'll print the results to console.
'''

import pandas as pd
from tabulate import tabulate
import os

def compute_judge_agreement_matrix_all(data: pd.DataFrame, save_to_file: str = None) -> None:
    """
    Compute the judge agreement matrix for all judgments.
    For "all", it refers to "w/ C" to distinguish from "w/o C".
    That is to say, judgments like {C,A} or {B,C} are considered disagreed.

    For Option-2 mode, like for DevBench, it doesn't matter which one to use and they will give same results.

    Args:
        data (pd.DataFrame): The input dataframe containing the judge columns with extracted choices.
        save_to_file (str, optional): The path to save the agreement matrix as a CSV file.
            If None, the matrix will be printed in the terminal. Defaults to None.

    Returns:
        None
    """
    # Get the list of judge columns
    judge_columns = [col for col in data.columns if col.startswith("extracted_")]
    
    # Extract the judge names from the column names
    judge_names = [col.replace("extracted_", "") for col in judge_columns]
    
    # Initialize the agreement matrix
    agreement_matrix = pd.DataFrame(columns=judge_names, index=judge_names)
    
    # Iterate over each pair of judges
    for judge1, judge1_col in zip(judge_names, judge_columns):
        for judge2, judge2_col in zip(judge_names, judge_columns):
            agree_count = 0
            total_count = 0
            
            # Iterate over each row in the dataframe
            for _, row in data.iterrows():
                if row[judge1_col] == row[judge2_col] and row[judge1_col] != 'N':
                    agree_count += 1
                if row[judge1_col] != 'N' and row[judge2_col] != 'N':
                    total_count += 1
            
            # Calculate the agreement percentage
            agreement = agree_count / total_count if total_count > 0 else 0
            
            # Store the agreement in the matrix
            agreement_matrix.loc[judge1, judge2] = agreement
    
    # Format the agreement matrix for better readability
    formatted_matrix = agreement_matrix.map(lambda x: f"{x:.2%}")
    
    if save_to_file:
        directory = os.path.dirname(save_to_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the agreement matrix to a CSV file
        formatted_matrix.to_csv(save_to_file)
    else:
        # Print the agreement matrix in a beautiful way in the terminal
        print(tabulate(formatted_matrix, headers='keys', tablefmt='psql'))
    print("compute_judge_agreement_matrix_all done.")


def compute_judge_agreement_matrix_without_C(data: pd.DataFrame, save_to_file: str = None) -> None:
    """
    Compute the judge agreement matrix excluding judgments 'C' and 'N'.
    The motivation for this that the extent of disagreement is smaller for judgments like {C,A} compared to {B,A}.

    Args:
        data (pd.DataFrame): The input dataframe containing the judge columns with extracted choices.
        save_to_file (str, optional): The path to save the agreement matrix as a CSV file.
            If None, the matrix will be printed in the terminal. Defaults to None.

    Returns:
        None
    """
    # Get the list of judge columns
    judge_columns = [col for col in data.columns if col.startswith("extracted_")]
    
    # Extract the judge names from the column names
    judge_names = [col.replace("extracted_", "") for col in judge_columns]
    
    # Initialize the agreement matrix
    agreement_matrix = pd.DataFrame(columns=judge_names, index=judge_names)
    
    # Iterate over each pair of judges
    for judge1, judge1_col in zip(judge_names, judge_columns):
        for judge2, judge2_col in zip(judge_names, judge_columns):
            agree_count = 0
            total_count = 0
            
            # Iterate over each row in the dataframe
            for _, row in data.iterrows():
                if row[judge1_col] == row[judge2_col] and row[judge1_col] not in ['C', 'N']:
                    agree_count += 1
                if row[judge1_col] not in ['C', 'N'] and row[judge2_col] not in ['C', 'N']:
                    total_count += 1
            
            # Calculate the agreement percentage
            agreement = agree_count / total_count if total_count > 0 else 0
            
            # Store the agreement in the matrix
            agreement_matrix.loc[judge1, judge2] = agreement
    
    # Format the agreement matrix for better readability
    formatted_matrix = agreement_matrix.map(lambda x: f"{x:.2%}")
    
    if save_to_file:
        directory = os.path.dirname(save_to_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the agreement matrix to a CSV file
        formatted_matrix.to_csv(save_to_file)
    else:
        # Print the agreement matrix in a beautiful way in the terminal
        print(tabulate(formatted_matrix, headers='keys', tablefmt='psql'))
    print("compute_judge_agreement_matrix_without_C done.")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_judge_agreement(agreement_matrix, order, graph_name, save_to_file=None):
    """
    Plot a heatmap of the agreement matrix between judge models.
    
    Args:
        agreement_matrix (pd.DataFrame): A dataframe with rows and columns being judge model names.
        order (list): A list of judge model names to display in the agreement matrix in specified order.
        graph_name (str): The title of the graph.
        save_to_file (str, optional): Path to save the heatmap image. If None, the heatmap is displayed without saving.
    
    Returns:
        None
    """
    # Set the index of the agreement_matrix to the first column
    agreement_matrix = agreement_matrix.set_index(agreement_matrix.columns[0])
    
    # Convert percentage values to floats
    agreement_matrix = agreement_matrix.map(lambda x: float(x[:-1]))
    
    # Reorder the agreement_matrix based on the provided order
    agreement_matrix = agreement_matrix.reindex(index=order, columns=order)
    
    # Create a new figure and set the figure size
    plt.figure(figsize=(12, 9))
    
    # Create the heatmap using seaborn
    # For more colors, see this link: https://seaborn.pydata.org/tutorial/color_palettes.html
    sns.heatmap(agreement_matrix, annot=True, cmap='mako', cbar_kws={'label': 'Agreement'}, fmt='.2f', annot_kws={'size': 14})
    
    # Set the plot title using the provided graph_name
    plt.title(graph_name, fontsize=20)
    
    # Set the x-axis and y-axis labels
    plt.xlabel('Judge Models', fontsize=16)
    plt.ylabel('Judge Models', fontsize=16)
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)
    
    # Adjust the y-axis tick label font size
    plt.yticks(fontsize=14)
    
    # Adjust the plot layout to prevent overlapping labels
    plt.tight_layout()
    
    # Save the heatmap to the specified path if provided
    if save_to_file:
        plt.savefig(save_to_file)
    else:
        # Display the heatmap
        plt.show()
        
if __name__=='__main__':
    # You'll need to first aggregate data and extract answers.

    MTBench_data = pd.read_csv('path/to/aggregated_data.csv')
    DevBench_data = pd.read_csv('path/to/aggregated_data.csv')

    compute_judge_agreement_matrix_all(MTBench_data,save_to_file="MTBench/judge_agreement/MTBench_judge_agreement_all.csv")
    compute_judge_agreement_matrix_without_C(MTBench_data, save_to_file="MTBench/judge_agreement/MTBench_judge_agreement_without_C.csv")
    compute_judge_agreement_matrix_all(DevBench_data, save_to_file="DevBench/judge_agreement/DevBench_judge_agreement_all.csv")

    MTBench_agreement_matrix_all = pd.read_csv('MTBench/judge_agreement/MTBench_judge_agreement_all.csv')
    MTBench_agreement_matrix_without_C = pd.read_csv('MTBench/judge_agreement/MTBench_judge_agreement_without_C.csv')
    DevBench_agreement_matrix_all = pd.read_csv('DevBench/judge_agreement/DevBench_judge_agreement_all.csv')

    order = [
        'gpt-4-0125-preview',
        'gpt-4-1106-preview',
        'gpt-4-0613',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307',
        'gemini-pro'
    ]
    plot_judge_agreement(agreement_matrix=MTBench_agreement_matrix_all,
                         order=order,
                         graph_name="MTBench Judge Agreement (All) Heatmap",
                         save_to_file='MTBench/judge_agreement/judge_agreement_all.png')
    
    plot_judge_agreement(agreement_matrix=MTBench_agreement_matrix_without_C,
                         order=order,
                         graph_name="MTBench Judge Agreement (without C) Heatmap",
                         save_to_file='MTBench/judge_agreement/judge_agreement_without_C.png')
    
    plot_judge_agreement(agreement_matrix=DevBench_agreement_matrix_all,
                        order=order,
                        graph_name="DevBench Judge Agreement (All) Heatmap",
                        save_to_file='DevBench/judge_agreement/judge_agreement_all.png')