#!/usr/bin/env python3
"""
Analysis and visualization of ACT policy performance on can manipulation tasks.
This script reads results from can_policies.csv and creates comparative visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import argparse
import os
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
import warnings


def parse_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Parse the complex CSV format with multiple policies and trials.
    
    Args:
        csv_path: Path to the can_policies.csv file
        
    Returns:
        Cleaned DataFrame with columns: Policy, Trial, Color, Task, Score
    """
    # Read raw CSV
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)
    
    # Define the subtasks we're tracking
    subtasks = [
        "Hand Move to Can",
        "Hand Grasp Can", 
        "Hand Move to correct Box",
        "Can in correct Box"
    ]
    
    # Initialize list to store parsed data
    parsed_data = []
    
    # Process each policy section
    policy_start_rows = []
    for idx, row in raw_df.iterrows():
        if pd.notna(row[0]) and any(policy in str(row[0]).lower() for policy in ['cans_resnet', 'dino']):
            policy_start_rows.append(idx)
    
    for policy_idx, start_row in enumerate(policy_start_rows):
        # Extract policy name
        policy_name = str(raw_df.iloc[start_row, 0]).replace('cans_', '').replace('_', ' ').title()
        
        # Find the end of this policy section
        end_row = policy_start_rows[policy_idx + 1] if policy_idx + 1 < len(policy_start_rows) else len(raw_df)
        
        # Extract color sequence (first row after policy name)
        color_row = raw_df.iloc[start_row, 1:]
        colors = [str(c).lower() if pd.notna(c) and str(c).lower() in ['red', 'green', 'black'] else None 
                 for c in color_row]
        
        # Count trials (number of color entries / 8, since each trial has 8 columns including 'end' and 'time')
        valid_colors = [c for c in colors if c is not None]
        n_trials = len(valid_colors) // 6  # 6 colors per trial
        
        # Process each subtask
        for task_offset, task_name in enumerate(subtasks):
            task_row_idx = start_row + 1 + task_offset
            if task_row_idx >= end_row:
                continue
                
            task_row = raw_df.iloc[task_row_idx, 1:]
            
            # Extract scores for each trial
            trial_idx = 0
            for col_idx in range(0, len(task_row), 8):  # Every 8 columns is a new trial
                if trial_idx >= n_trials:
                    break
                    
                # Get the 6 scores for this trial (excluding 'end' and 'time' columns)
                trial_scores = []
                for score_idx in range(6):
                    if col_idx + score_idx < len(task_row):
                        score = task_row.iloc[col_idx + score_idx]
                        if pd.notna(score) and str(score) not in ['None', 'end', 'time', '']:
                            try:
                                trial_scores.append(float(score))
                            except ValueError:
                                pass
                
                # Calculate mean score for this trial and task
                if trial_scores:
                    mean_score = np.mean(trial_scores)
                    
                    # Get trial color (every 6 colors is a new trial)
                    color_idx = trial_idx * 6
                    trial_color = valid_colors[color_idx] if color_idx < len(valid_colors) else 'unknown'
                    
                    parsed_data.append({
                        'Policy': policy_name,
                        'Trial': trial_idx + 1,
                        'Color': trial_color,
                        'Task': task_name,
                        'Score': mean_score
                    })
                
                trial_idx += 1
    
    return pd.DataFrame(parsed_data)


def create_radar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a radar chart comparing all policies across subtasks."""
    
    # Calculate mean scores per policy and task
    policy_stats = df.groupby(['Policy', 'Task'])['Score'].mean().unstack(fill_value=0)
    
    # Ensure all subtasks are present
    subtasks = ["Hand Move to Can", "Hand Grasp Can", "Hand Move to correct Box", "Can in correct Box"]
    for task in subtasks:
        if task not in policy_stats.columns:
            policy_stats[task] = 0
    
    policy_stats = policy_stats[subtasks]  # Reorder columns
    
    # Set up radar chart
    N = len(subtasks)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Plot each policy
    colors = plt.cm.Set1(np.linspace(0, 1, len(policy_stats)))
    for idx, (policy, scores) in enumerate(policy_stats.iterrows()):
        values = scores.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=policy, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Customize the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subtasks, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('Sort Cans Policy Performance', size=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_chart_policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_grouped_bar_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create grouped bar plots with error bars for each subtask."""
    
    # Calculate statistics
    stats = df.groupby(['Policy', 'Task'])['Score'].agg(['mean', 'std', 'count']).reset_index()
    
    subtasks = ["Hand Move to Can", "Hand Grasp Can", "Hand Move to correct Box", "Can in correct Box"]
    policies = stats['Policy'].unique()
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create a subplot for each subtask
    for task_idx, task in enumerate(subtasks):
        ax = axes[task_idx]
        
        task_data = stats[stats['Task'] == task]
        
        x = np.arange(len(policies))
        means = []
        stds = []
        
        for policy in policies:
            policy_data = task_data[task_data['Policy'] == policy]
            if len(policy_data) > 0:
                means.append(policy_data['mean'].iloc[0])
                stds.append(policy_data['std'].iloc[0] if pd.notna(policy_data['std'].iloc[0]) else 0)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bars with error bars
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=plt.cm.Set2(np.linspace(0, 1, len(policies))))
        
        # Customize subplot
        ax.set_title(f'{task}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_xlabel('Policy', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Sort Cans Policy Performance)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_statistical_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Perform comprehensive statistical analysis on the policy performance data."""
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    results = {}
    
    print("\nPerforming statistical analysis...")
    print("=" * 60)
    
    # 1. ANOVA: Policy differences across all tasks
    print("\n1. POLICY PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    policy_anova_results = {}
    for task in df['Task'].unique():
        task_data = df[df['Task'] == task]
        policy_groups = [task_data[task_data['Policy'] == policy]['Score'].values 
                        for policy in task_data['Policy'].unique()]
        
        # Remove empty groups
        policy_groups = [group for group in policy_groups if len(group) > 0]
        
        if len(policy_groups) >= 2:
            try:
                f_stat, p_value = f_oneway(*policy_groups)
                policy_anova_results[task] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"{task}:")
                print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.4f} {significance}")
            except Exception as e:
                print(f"{task}: Could not perform ANOVA - {str(e)}")
    
    results['policy_anova'] = policy_anova_results
    
    # 2. Color effect analysis
    print("\n2. COLOR EFFECT ANALYSIS")
    print("-" * 40)
    
    color_results = {}
    colors_available = df['Color'].unique()
    colors_available = [c for c in colors_available if c not in ['unknown', None]]
    
    if len(colors_available) >= 2:
        for task in df['Task'].unique():
            task_data = df[df['Task'] == task]
            color_groups = [task_data[task_data['Color'] == color]['Score'].values 
                           for color in colors_available]
            
            # Remove empty groups
            color_groups = [group for group in color_groups if len(group) > 0]
            
            if len(color_groups) >= 2:
                try:
                    f_stat, p_value = f_oneway(*color_groups)
                    color_results[task] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'colors_tested': colors_available
                    }
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"{task}:")
                    print(f"  Colors tested: {colors_available}")
                    print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.4f} {significance}")
                    
                    # Pairwise t-tests for significant results
                    if p_value < 0.05 and len(color_groups) == 2:
                        t_stat, t_p = ttest_ind(color_groups[0], color_groups[1])
                        print(f"  Pairwise t-test: t={t_stat:.3f}, p={t_p:.4f}")
                        
                except Exception as e:
                    print(f"{task}: Could not perform color analysis - {str(e)}")
    else:
        print("Insufficient color data for analysis")
    
    results['color_analysis'] = color_results
    
    # 3. Policy-specific color effects
    print("\n3. POLICY-SPECIFIC COLOR EFFECTS")
    print("-" * 40)
    
    policy_color_results = {}
    for policy in df['Policy'].unique():
        policy_data = df[df['Policy'] == policy]
        
        print(f"\nPolicy: {policy}")
        policy_color_results[policy] = {}
        
        for task in df['Task'].unique():
            task_data = policy_data[policy_data['Task'] == task]
            colors_in_task = task_data['Color'].unique()
            colors_in_task = [c for c in colors_in_task if c not in ['unknown', None]]
            
            if len(colors_in_task) >= 2:
                color_groups = [task_data[task_data['Color'] == color]['Score'].values 
                               for color in colors_in_task]
                color_groups = [group for group in color_groups if len(group) > 0]
                
                if len(color_groups) >= 2:
                    try:
                        if len(color_groups) == 2:
                            t_stat, p_value = ttest_ind(color_groups[0], color_groups[1])
                            stat_name = "t-statistic"
                            stat_value = t_stat
                        else:
                            stat_value, p_value = f_oneway(*color_groups)
                            stat_name = "F-statistic"
                        
                        policy_color_results[policy][task] = {
                            'statistic': stat_value,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'colors': colors_in_task
                        }
                        
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        print(f"  {task}: {stat_name}={stat_value:.3f}, p={p_value:.4f} {significance}")
                        
                    except Exception as e:
                        print(f"  {task}: Could not analyze - {str(e)}")
    
    results['policy_color_analysis'] = policy_color_results
    
    # 4. Effect sizes (Cohen's d for pairwise comparisons)
    print("\n4. EFFECT SIZES (Cohen's d)")
    print("-" * 40)
    
    effect_sizes = {}
    for task in df['Task'].unique():
        task_data = df[df['Task'] == task]
        policies = task_data['Policy'].unique()
        
        if len(policies) >= 2:
            effect_sizes[task] = {}
            
            # Calculate Cohen's d for all policy pairs
            for i, policy1 in enumerate(policies):
                for policy2 in policies[i+1:]:
                    group1 = task_data[task_data['Policy'] == policy1]['Score'].values
                    group2 = task_data[task_data['Policy'] == policy2]['Score'].values
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # Cohen's d calculation
                        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                            (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                           (len(group1) + len(group2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                            
                            # Interpret effect size
                            if abs(cohens_d) < 0.2:
                                interpretation = "negligible"
                            elif abs(cohens_d) < 0.5:
                                interpretation = "small"
                            elif abs(cohens_d) < 0.8:
                                interpretation = "medium"
                            else:
                                interpretation = "large"
                            
                            effect_sizes[task][f"{policy1} vs {policy2}"] = {
                                'cohens_d': cohens_d,
                                'interpretation': interpretation
                            }
                            
                            print(f"{task} - {policy1} vs {policy2}:")
                            print(f"  Cohen's d: {cohens_d:.3f} ({interpretation})")
    
    results['effect_sizes'] = effect_sizes
    
    # 5. Success rate distributions
    print("\n5. SUCCESS RATE DISTRIBUTIONS")
    print("-" * 40)
    
    distribution_analysis = {}
    for policy in df['Policy'].unique():
        policy_data = df[df['Policy'] == policy]
        scores = policy_data['Score'].values
        
        if len(scores) > 0:
            # Shapiro-Wilk test for normality
            try:
                shapiro_stat, shapiro_p = stats.shapiro(scores)
                is_normal = shapiro_p > 0.05
                
                distribution_analysis[policy] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores, ddof=1),
                    'median': np.median(scores),
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': is_normal,
                    'n_samples': len(scores)
                }
                
                normality = "normal" if is_normal else "non-normal"
                print(f"{policy}:")
                print(f"  Mean: {np.mean(scores):.3f} ± {np.std(scores, ddof=1):.3f}")
                print(f"  Median: {np.median(scores):.3f}")
                print(f"  Distribution: {normality} (Shapiro-Wilk p={shapiro_p:.4f})")
                
            except Exception as e:
                print(f"{policy}: Could not analyze distribution - {str(e)}")
    
    results['distribution_analysis'] = distribution_analysis
    
    # Save detailed results to file
    with open(output_dir / 'statistical_analysis.txt', 'w') as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY OF FINDINGS:\n")
        f.write("-" * 40 + "\n")
        
        # Policy differences
        f.write("\n1. POLICY PERFORMANCE DIFFERENCES:\n")
        significant_tasks = [task for task, result in policy_anova_results.items() 
                           if result.get('significant', False)]
        if significant_tasks:
            f.write(f"   Significant policy differences found in: {', '.join(significant_tasks)}\n")
        else:
            f.write("   No significant policy differences found.\n")
        
        # Color effects
        f.write("\n2. COLOR EFFECTS:\n")
        significant_color_tasks = [task for task, result in color_results.items() 
                                 if result.get('significant', False)]
        if significant_color_tasks:
            f.write(f"   Significant color effects found in: {', '.join(significant_color_tasks)}\n")
        else:
            f.write("   No significant color effects found.\n")
        
        # Large effect sizes
        f.write("\n3. LARGE EFFECT SIZES (Cohen's d > 0.8):\n")
        large_effects = []
        for task, comparisons in effect_sizes.items():
            for comparison, result in comparisons.items():
                if abs(result['cohens_d']) > 0.8:
                    large_effects.append(f"{task}: {comparison} (d={result['cohens_d']:.2f})")
        
        if large_effects:
            for effect in large_effects:
                f.write(f"   {effect}\n")
        else:
            f.write("   No large effect sizes found.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS:\n\n")
        
        # Write detailed results
        import json
        f.write(json.dumps(results, indent=2, default=str))
    
    print(f"\nDetailed statistical analysis saved to: {output_dir / 'statistical_analysis.txt'}")
    
    # Return results for potential further use
    return results


def create_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save summary statistics."""
    
    # Overall statistics by policy
    overall_stats = df.groupby('Policy')['Score'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    # Statistics by policy and task
    task_stats = df.groupby(['Policy', 'Task'])['Score'].agg([
        'count', 'mean', 'std'
    ]).round(3)
    
    # Color-based analysis
    color_stats = df.groupby(['Policy', 'Color'])['Score'].agg([
        'count', 'mean', 'std'
    ]).round(3)
    
    # Save statistics to file
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("CAN MANIPULATION POLICY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. OVERALL PERFORMANCE BY POLICY\n")
        f.write("-" * 35 + "\n")
        f.write(overall_stats.to_string())
        f.write("\n\n")
        
        f.write("2. PERFORMANCE BY POLICY AND TASK\n")
        f.write("-" * 35 + "\n")
        f.write(task_stats.to_string())
        f.write("\n\n")
        
        f.write("3. PERFORMANCE BY POLICY AND COLOR\n")
        f.write("-" * 35 + "\n")
        f.write(color_stats.to_string())
        f.write("\n\n")
        
        # Best performing policy per task
        f.write("4. BEST PERFORMING POLICY PER TASK\n")
        f.write("-" * 35 + "\n")
        best_per_task = df.groupby(['Task', 'Policy'])['Score'].mean().unstack().idxmax(axis=1)
        for task, best_policy in best_per_task.items():
            best_score = df.groupby(['Task', 'Policy'])['Score'].mean().unstack().loc[task, best_policy]
            f.write(f"{task}: {best_policy} (Score: {best_score:.3f})\n")
    
    print(f"Summary statistics saved to: {output_dir / 'summary_statistics.txt'}")
    
    # Print key findings
    print("\nKEY FINDINGS:")
    print("=" * 50)
    print("\nOverall Performance by Policy:")
    print(overall_stats['mean'].sort_values(ascending=False))
    
    print("\nBest Policy per Task:")
    for task, best_policy in best_per_task.items():
        best_score = df.groupby(['Task', 'Policy'])['Score'].mean().unstack().loc[task, best_policy]
        print(f"  {task}: {best_policy} ({best_score:.3f})")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize ACT policy performance on can manipulation tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path", "-c",
        type=str,
        default="plot_wandb/can_policies.csv",
        help="Path to the CSV file containing policy test results"
    )
    
    parser.add_argument(
        "--output_dir", "-o", 
        type=str,
        default="plot_wandb/plots/can_analysis",
        help="Directory to save output plots and statistics"
    )
    
    parser.add_argument(
        "--plots",
        nargs="*",
        choices=["radar", "bars", "summary", "statistics", "all"],
        default=["all"],
        help="Choose which plots to create"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return 1
    
    print(f"Loading data from {args.csv_path}...")
    
    # Parse the CSV data
    try:
        df = parse_csv_data(args.csv_path)
        print(f"Successfully parsed {len(df)} data points")
        print(f"Policies found: {df['Policy'].unique()}")
        print(f"Tasks found: {df['Task'].unique()}")
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return 1
    
    # Save parsed data for inspection
    df.to_csv(output_dir / 'parsed_data.csv', index=False)
    print(f"Parsed data saved to: {output_dir / 'parsed_data.csv'}")
    
    # Determine which plots to create
    selected_plots = args.plots
    if "all" in selected_plots:
        selected_plots = ["radar", "bars", "summary", "statistics"]
    
    # Set plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # Create plots
    if "radar" in selected_plots:
        print("\nCreating radar chart...")
        create_radar_chart(df, output_dir)
    
    if "bars" in selected_plots:
        print("Creating grouped bar plots...")
        create_grouped_bar_plot(df, output_dir)
    
    if "summary" in selected_plots:
        print("Generating summary statistics...")
        create_summary_statistics(df, output_dir)
    
    if "statistics" in selected_plots:
        print("Performing statistical analysis...")
        perform_statistical_analysis(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
