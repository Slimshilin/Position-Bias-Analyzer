'''
This function visualizes the judge cumulative disagreement.

The input data should have choices extracted (i.e., should have columns named as extracted_{judge})

save_to_directory, when specified, would save the graphs to it (without showing); otherwise, it'll show the graphs.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_cumulative_judge_disagreement(data, graph_title, benchmark, save_to_directory=None):
    """
    Analyzes the judge disagreement for a given benchmark and creates a bar chart with a cumulative curve.
    Judge disagreement is the number of disagreed judgments compared to the mode on the same query instance.

    Args:
        data (pd.DataFrame): The input DataFrame containing the extracted columns.
        graph_title (str): The title of the graph.
        benchmark (str): The name of the benchmark used for saving files.
        save_to_directory (str, optional): The directory where the output files will be saved. If None, the chart will be displayed. Defaults to None.

    Returns:
        None
    """
    # Filter columns starting with 'extracted_'
    extracted_cols = [col for col in data.columns if col.startswith('extracted_')]
    
    # Compute disagreement for each row
    disagreements = []
    for _, row in data[extracted_cols].iterrows():
        # Exclude 'N' values and compute the mode
        valid_values = [val for val in row if val != 'N']
        if valid_values:
            mode = max(set(valid_values), key=valid_values.count)
            # Count values different from the mode
            disagreement = sum(val != mode for val in valid_values)
        else:
            disagreement = 0
        disagreements.append(disagreement)
    
    # Create a DataFrame with disagreement counts
    disagreement_counts = pd.DataFrame({'Disagreement': disagreements})
    disagreement_counts = disagreement_counts['Disagreement'].value_counts().reset_index()
    disagreement_counts.columns = ['Disagreement', 'Count']
    disagreement_counts = disagreement_counts.sort_values('Disagreement')
    
    # Save the disagreement_counts DataFrame to CSV if save_to_directory is specified
    if save_to_directory:
        if not os.path.exists(save_to_directory):
            os.makedirs(save_to_directory)
        
        csv_file = f"{save_to_directory}/{benchmark}-judge_disagreement_counts.csv"
        disagreement_counts.to_csv(csv_file, index=False)
        print(f"judge disagreement counts dataframe saved to {csv_file}")
    
    # Compute cumulative counts and percentages
    cumulative_counts = disagreement_counts['Count'].cumsum()
    total_count = cumulative_counts.iloc[-1]
    percentages = cumulative_counts / total_count * 100
    
    # Create the bar chart with cumulative curve and horizontal lines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(disagreement_counts['Disagreement'], disagreement_counts['Count'], color='#27aeef')

    ax.plot(disagreement_counts['Disagreement'], cumulative_counts, 'r.-', markersize=10)

    for x, y, p in zip(disagreement_counts['Disagreement'], cumulative_counts, percentages):
        ax.hlines(y, -0.5, x, linestyles='dashed', colors='#cccccc', linewidth=0.8)
        ax.annotate(f"{y} ({p:.1f}%)", xy=(x, y), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom')

    # The axis labels, revise according to your need.
    ax.set_xlabel('Disagreement (9 Judges)')
    ax.set_ylabel('Count of disagreed judgments')
    ax.set_title(graph_title)

    # Adjust the ylim to accommodate the annotations
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    
    # Save or display the chart
    if save_to_directory:
        plt.savefig(f"{save_to_directory}/{benchmark}-judge_disagreement_analysis.png")
        print(f"judge cumulative disagreement analysis graph saved to {save_to_directory}/{benchmark}-judge_disagreement_analysis.png")
    else:
        plt.show()

                
if __name__=="__main__":
    # You'll need to first aggregate data and extract answers.

    MTBench_data = pd.read_csv('path/to/aggregated_data.csv')
    DevBench_data = pd.read_csv('path/to/aggregated_data.csv')

    analyze_cumulative_judge_disagreement(data=MTBench_data,
                                          graph_title="MTBench Disagreement Analysis",
                                          benchmark="MTBench",
                                          save_to_directory="./MTBench")
    analyze_cumulative_judge_disagreement(data=DevBench_data,
                                        graph_title="DevBench Disagreement Analysis",
                                        benchmark="DevBench",
                                        save_to_directory="./DevBench")