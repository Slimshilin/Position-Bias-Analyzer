'''
This is the main function for running all analysis on Position Bias.

If you are using different number or type of judges, please revise the following things in the corresponding import function:
    - input_pricing_map
    - output_pricing_map
    - context_window
    - max_output_length
    - family_map
    - order (in this main function)

For analysis on a combination of datasets/benchmarks, please combine them manually and call the individual analysis functions.

Be consistent with your benchmark and nopt (Option mode, i.e., 2 or 3) input. Now most of the analysis function only support analysis for MTBencha and DevBench.

For repetitional consistency calculation, please visit the corresponding module, as the data preparation process is different.
'''


import os
import pandas as pd

# Judge Agreement & Disagreement 
from judging_the_judges.judge_agreement.compute_judge_agreement import compute_judge_agreement_matrix_all, compute_judge_agreement_matrix_without_C, plot_judge_agreement
from judging_the_judges.judge_agreement.analyze_cumulative_disagreement import analyze_cumulative_judge_disagreement

# Calculate Positional Consistency and Positional Preference Score
from judging_the_judges.calculate_consistency_and_preference.consistency_preference_util import aggregate_judge_data, extract_answer
from judging_the_judges.calculate_consistency_and_preference.consistency_preference_calculation import calculate_positional_consistency_and_preference_score

# Analysis based on purely Positional Consistency and Positional Preference Score
from judging_the_judges.judge_task_comparison.plot_baseline_comparison_graph import plot_baseline_positional_consistency_comparison_graph, plot_baseline_positional_preference_score_comparison_graph
from judging_the_judges.judge_task_comparison.by_Judge_Task_and_Task_Judge import plot_positional_consistency_and_preference_score_by_Judge_Task, plot_positional_consistency_and_preference_score_by_Task_Judge
from judging_the_judges.judge_task_comparison.MLE_analysis import MLE_each_judge,MLE_each_task_each_judge

from judging_the_judges.position_bias_vs_factors.analyze_positional_consistency_vs_preference_score import analyze_positional_consistency_vs_preference_score_linear

# Position bias vs. Other factors
## Calculation and Augmentation (Preparation)
from judging_the_judges.position_bias_vs_factors.position_bias_vs_winrates.calculate_consistent_winrate import calculate_consistent_winrate, augment_results_with_consistent_winrates
from judging_the_judges.position_bias_vs_factors.position_bias_vs_winrates.calculate_overall_winrate import calculate_overall_winrate, augment_results_with_overall_winrates
from judging_the_judges.position_bias_vs_factors.position_bias_vs_lengths.calculate_length_stats import calculate_length_stats,augment_length_stats

## Analysis
from judging_the_judges.position_bias_vs_factors.position_bias_vs_winrates.analyze_position_bias_vs_winrates import analyze_positional_consistency_and_preference_score_vs_winrates_nonlinear
from judging_the_judges.position_bias_vs_factors.position_bias_vs_lengths.analyze_position_bias_vs_lengths import analyze_positional_consistency_and_preference_score_vs_log_length_stat_nonlinear

# Linear Regression
from judging_the_judges.position_bias_vs_factors.linear_regression.prepare_linear_regression_data import prepare_linear_regression_data,context_window,max_output_length,family_map
from judging_the_judges.position_bias_vs_factors.linear_regression.linear_regression import run_linear_regression

# Price-Performance Ratio 
from judging_the_judges.judge_task_comparison.price_vs_consistency import plot_judge_level_price_vs_positional_consistency, plot_judge_task_level_price_vs_positional_consistency, input_pricing_map, output_pricing_map

def judging_position_bias(data_directory, benchmark, nopt, save_to_directory, input_pricing_map=input_pricing_map, output_pricing_map=output_pricing_map, context_window=context_window, max_output_length=max_output_length, family_map=family_map):
    """
    Main function for analyzing position bias of judgments.
    
    Data Preparation: 
    - Put all resulting judgment output folders in a single directory
    - Do not change the original file names of resulting files
    
    The function does the following:
    1. Compute judge agreement matrices with and without C (ties)
    2. Analyze cumulative judge disagreement 
    3. Calculate positional consistency and preference scores
    4. Plot baseline comparisons for positional consistency and preference scores
    5. Plot positional consistency and preference scores by judge-task and task-judge
    6. Perform MLE analysis for each judge and each task-judge
    7. Analyze positional consistency vs preference score
    8. Calculate and augment results with overall and consistent winrates and length statistics
    9. Analyze position bias vs winrates and lengths
    10. Prepare data for and perform linear regression
    11. Plot price vs positional consistency at judge and judge-task levels
    """
    
    # Aggregate judge data and extract answers
    data = aggregate_judge_data(data_directory, nopt)
    data = extract_answer(data, benchmark)
    
    # Defining order for judge sort in subfigures
    order = ['gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4-0613', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106', 
             'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'gemini-pro']
    
    # 1. Compute judge agreement matrices with and without C (ties)
    print("="*20, "Judge Agreement Matrix", "="*20)
    compute_judge_agreement_matrix_all(data, save_to_file=os.path.join(save_to_directory, "judge_agreement", f"{benchmark}_judge_agreement_all.csv"))
    agreement_matrix_all = pd.read_csv(os.path.join(save_to_directory, "judge_agreement", f"{benchmark}_judge_agreement_all.csv"))
    plot_judge_agreement(agreement_matrix=agreement_matrix_all, order=order, graph_name=f"{benchmark} Judge Agreement (All) Heatmap", 
                         save_to_file=os.path.join(save_to_directory, "judge_agreement", "judge_agreement_all.png"))
    
    if nopt == 3:
        compute_judge_agreement_matrix_without_C(data, save_to_file=os.path.join(save_to_directory, "judge_agreement", f"{benchmark}_judge_agreement_without_C.csv"))
        agreement_matrix_without_C = pd.read_csv(os.path.join(save_to_directory, "judge_agreement", f"{benchmark}_judge_agreement_without_C.csv"))
        plot_judge_agreement(agreement_matrix=agreement_matrix_without_C, order=order, graph_name=f"{benchmark} Judge Agreement (without C) Heatmap", 
                             save_to_file=os.path.join(save_to_directory, "judge_agreement", "judge_agreement_without_C.png"))
    print("="*(42 + len("Judge Agreement Matrix")))
    print()
    
    # 2. Analyze cumulative judge disagreement
    print("="*20, "Cumulative Judge Disagreement", "="*20)
    analyze_cumulative_judge_disagreement(data=data, graph_title=f"{benchmark} Disagreement Analysis", benchmark=benchmark, 
                                          save_to_directory=os.path.join(save_to_directory))
    print("="*(42 + len("Cumulative Judge Disagreement")))
    print()
    
    # 3. Calculate positional consistency and preference scores  
    print("="*20, "Positional Consistency and Preference Scores", "="*20)
    results_df, averages_df = calculate_positional_consistency_and_preference_score(data, reference_model="vicuna-13b-v1.3" if benchmark=="MTBench" else "human")
    results_df.to_csv(os.path.join(save_to_directory, "(Judge-Model-Task)_results.csv"), index=False)
    averages_df.to_csv(os.path.join(save_to_directory, "Judge_average_results.csv"), index=False)
    print("="*(42 + len("Positional Consistency and Preference Scores")))
    print()
    
    # 4. Plot baseline comparisons for positional consistency and preference scores
    print("="*20, "Baseline Comparisons", "="*20)
    plot_baseline_positional_consistency_comparison_graph(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} Positional Consistency Baseline Comparison', 
                                                          baseline_judge='gpt-4-0613', save_to_directory=os.path.join(save_to_directory, 'baseline-comparison'))
    plot_baseline_positional_preference_score_comparison_graph(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} Positional Preference Score Baseline Comparison', 
                                                               save_to_directory=os.path.join(save_to_directory, 'baseline-comparison'))
    print("="*(42 + len("Baseline Comparisons")))
    print()
    
    # 5. Plot positional consistency and preference scores by judge-task and task-judge  
    print("="*20, "Positional Consistency and Preference Scores by Judge-Task and Task-Judge", "="*20)
    plot_positional_consistency_and_preference_score_by_Judge_Task(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} Positional Consistency \n By-Judge-Task', 
                                                                   save_to_directory=os.path.join(save_to_directory, 'By-Judge_and_Task'))
    plot_positional_consistency_and_preference_score_by_Task_Judge(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} Positional Preference Score \n By-Task-Judge', 
                                                                   save_to_directory=os.path.join(save_to_directory, 'By-Judge_and_Task'))
    print("="*(42 + len("Positional Consistency and Preference Scores by Judge-Task and Task-Judge")))
    print()
    
    # 6. Perform MLE analysis for each judge and each task-judge
    print("="*20, "MLE Analysis", "="*20)
    MLE_each_task_each_judge(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} MLE for Each Judge and Task', 
                             order=order, save_to_directory=os.path.join(save_to_directory,'MLE'))
    MLE_each_judge(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} MLE for Each Judge', 
                   save_to_directory=os.path.join(save_to_directory,'MLE'))
    print("="*(42 + len("MLE Analysis")))
    print()
    
    # 7. Analyze positional consistency vs preference score
    print("="*20, "Positional Consistency vs Preference Score", "="*20)
    analyze_positional_consistency_vs_preference_score_linear(results_df=results_df, benchmark=benchmark, graph_title=f'{benchmark} Positional Consistency vs. Positional Preference Score', 
                                                              save_to_directory=os.path.join(save_to_directory, "consistency_vs_preference"))
    print("="*(42 + len("Positional Consistency vs Preference Score")))
    print()
    
    # 8. Calculate and augment results with overall and consistent winrates and length statistics
    print("="*20, "Calculate Winrates and Length Statistics", "="*20)
    overall_winrate_df = calculate_overall_winrate(data=data, reference_model='vicuna-13b-v1.3' if benchmark=='MTBench' else 'human')
    consistent_winrate_df = calculate_consistent_winrate(data=data, reference_model='vicuna-13b-v1.3' if benchmark=='MTBench' else 'human')  
    length_stats_results_df = calculate_length_stats(data=data, benchmark=benchmark, reference_model='vicuna-13b-v1.3' if benchmark=='MTBench' else 'human')
    
    result_df_with_metrics = augment_results_with_overall_winrates(results_df, overall_winrate_df)
    result_df_with_metrics = augment_results_with_consistent_winrates(result_df_with_metrics, consistent_winrate_df)
    result_df_with_metrics = augment_length_stats(previous_results_df=result_df_with_metrics, length_results_df=length_stats_results_df)
    result_df_with_metrics.to_csv(os.path.join(save_to_directory, "result_df_with_metrics.csv"), index=False)
    print("="*(42 + len("Calculate Winrates and Length Statistics")))
    print()
    
    # 9. Analyze position bias vs winrates and lengths
    print("="*20, "Position Bias vs Winrates and Lengths", "="*20)
    analyze_positional_consistency_and_preference_score_vs_winrates_nonlinear(results_df=result_df_with_metrics, benchmark=benchmark, 
                                                                              save_to_directory=os.path.join(save_to_directory, 'winrate'))
    analyze_positional_consistency_and_preference_score_vs_log_length_stat_nonlinear(results_df=result_df_with_metrics, benchmark=benchmark, 
                                                                                      save_to_directory=os.path.join(save_to_directory, 'lengths'))
    print("="*(42 + len("Position Bias vs Winrates and Lengths")))
    print()
    
    # 10. Prepare data for and perform linear regression  
    print("="*20, "Linear Regression", "="*20)
    prepared_data = prepare_linear_regression_data(results_df=result_df_with_metrics, benchmark=benchmark, context_window=context_window,
                                                   max_output_length=max_output_length, family_map=family_map, 
                                                   save_to_directory=os.path.join(save_to_directory, 'linear_regression'))
    summary_consistency, summary_preference = run_linear_regression(data=prepared_data, benchmark=benchmark, dummy_included=True, 
                                                                    save_to_directory=os.path.join(save_to_directory, 'linear_regression'))
    print("="*(42 + len("Linear Regression")))
    print()
    
    # 11. Plot price vs positional consistency at judge and judge-task levels
    print("="*20, "Price vs Positional Consistency", "="*20)
    plot_judge_task_level_price_vs_positional_consistency(results_df=result_df_with_metrics, benchmark=benchmark, 
                                                          graph_title=f'{benchmark} Price vs. Positional Consistency (Task Level)', subplot_layout=(4,2) if benchmark=="MTBench" else (5,3),
                                                          input_pricing_map=input_pricing_map, output_pricing_map=output_pricing_map, 
                                                          save_to_directory=os.path.join(save_to_directory, "price_vs_consistency"))
    plot_judge_level_price_vs_positional_consistency(results_df=result_df_with_metrics, benchmark=benchmark,
                                                     graph_title=f'{benchmark} Price vs. Positional Consistency', 
                                                     input_pricing_map=input_pricing_map, output_pricing_map=output_pricing_map,
                                                     save_to_directory=os.path.join(save_to_directory, "price_vs_consistency"))
    print("="*(42 + len("Price vs Positional Consistency")))
    print()

if __name__ == "__main__":
    print("="*(102 + len("MTBench")))
    print("="*50, "MTBench", "="*50)
    print("="*(102 + len("MTBench")))
    judging_position_bias(data_directory='judgment_data/data_MTBench_position_bias', 
                          benchmark='MTBench', 
                          nopt=3,
                          save_to_directory='MTBench_analysis')
    
    print()
    print()
    print()

    print("="*(102 + len("DevBench")))
    print("="*50, "DevBench", "="*50)
    print("="*(102 + len("DevBench")))
    judging_position_bias(data_directory='judgment_data/data_DevBench_position_bias', 
                        benchmark='DevBench', 
                        nopt=2,
                        save_to_directory='DevBench_analysis')