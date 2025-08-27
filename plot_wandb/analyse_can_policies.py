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
    
    # Professional color scheme for thesis
    thesis_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Set up radar chart with better proportions
    N = len(subtasks)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'), dpi=150)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Enhanced grid styling
    ax.grid(True, alpha=0.4, linewidth=0.8, color='gray')
    ax.set_facecolor('#fafafa')
    
    # Plot each policy with enhanced styling
    for idx, (policy, scores) in enumerate(policy_stats.iterrows()):
        values = scores.tolist()
        values += values[:1]  # Complete the circle
        
        color = thesis_colors[idx % len(thesis_colors)]
        
        # Main line with enhanced styling
        ax.plot(angles, values, 'o-', linewidth=3, label=policy, color=color, 
                markersize=8, markerfacecolor=color, markeredgecolor='white', 
                markeredgewidth=2, alpha=0.9)
        
        # Semi-transparent fill
        ax.fill(angles, values, alpha=0.08, color=color)
        
        # Add value labels on points for better readability
        for angle, value in zip(angles[:-1], values[:-1]):
            if value > 0.05:  # Only show labels for non-zero values
                ax.text(angle, value + 0.05, f'{value:.2f}', 
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor=color, alpha=0.8))
    
    # Enhanced axis customization
    ax.set_xticks(angles[:-1])
    
    # Better task labels with line breaks for readability
    task_labels = [
        "Move to\nCan",
        "Grasp\nCan", 
        "Move to\nCorrect Box",
        "Place Can in\nCorrect Box"
    ]
    ax.set_xticklabels(task_labels, fontsize=12, fontweight='bold', ha='center')
    
    # Enhanced radial axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                       fontsize=11, alpha=0.8, fontweight='medium')
    
    # Add radial grid lines at specific values
    for tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot([0, 2*pi], [tick, tick], color='gray', alpha=0.3, linewidth=0.8)
    
    # Professional title and styling
    plt.title('Policy Performance Comparison on Can Sorting Task', 
              size=18, fontweight='bold', pad=40, color='#2c3e50')
    
    # Enhanced legend
    legend = ax.legend(loc='center', bbox_to_anchor=(1.4, 0.5), 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=12, title='ACT Policies', title_fontsize=14)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    
    # Add subtitle for context
    fig.text(0.5, 0.92, 'Success Rate by Subtask (0.0 = Failure, 1.0 = Perfect Success)', 
             ha='center', va='top', fontsize=12, style='italic', color='#6c757d')
    
    # Professional layout
    plt.tight_layout()
    
    # Save with multiple formats for thesis use
    plt.savefig(output_dir / 'radar_chart_policy_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'radar_chart_policy_comparison.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'radar_chart_policy_comparison.svg', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.show()


def create_grouped_bar_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create beautiful grouped bar plots with error bars for each subtask."""
    
    # Calculate statistics
    stats = df.groupby(['Policy', 'Task'])['Score'].agg(['mean', 'std', 'count']).reset_index()
    
    subtasks = ["Hand Move to Can", "Hand Grasp Can", "Hand Move to correct Box", "Can in correct Box"]
    policies = stats['Policy'].unique()
    
    # Professional color scheme for thesis
    thesis_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Set up the plot with better spacing and professional styling
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    axes = axes.flatten()
    
    # Global styling
    fig.patch.set_facecolor('white')
    
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
                std_val = policy_data['std'].iloc[0] if pd.notna(policy_data['std'].iloc[0]) else 0
                stds.append(std_val)
            else:
                means.append(0)
                stds.append(0)
        
        # Create beautiful bars with enhanced styling
        bars = ax.bar(x, means, yerr=stds, capsize=8, capthick=2,
                     color=[thesis_colors[i % len(thesis_colors)] for i in range(len(policies))],
                     alpha=0.85, edgecolor='white', linewidth=2,
                     error_kw={'elinewidth': 2, 'ecolor': '#2c3e50', 'alpha': 0.8})
        
        # Add gradient effect to bars
        for i, bar in enumerate(bars):
            # Add subtle gradient by varying alpha
            gradient = plt.Rectangle((bar.get_x(), 0), bar.get_width(), bar.get_height(),
                                   facecolor=thesis_colors[i % len(thesis_colors)], 
                                   alpha=0.3, edgecolor='none')
            ax.add_patch(gradient)
        
        # Enhanced subplot styling
        ax.set_facecolor('#fafafa')
        ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=1, color='#bdc3c7')
        ax.set_axisbelow(True)
        
        # Customize subplot titles with better formatting
        task_title = task.replace(' to ', ' to\n') if len(task) > 20 else task
        ax.set_title(f'{task_title}', fontsize=16, fontweight='bold', 
                    pad=15, color='#2c3e50')
        
        # Enhanced axis labels
        ax.set_ylabel('Success Rate', fontsize=14, fontweight='medium', color='#2c3e50')
        ax.set_xlabel('Policy', fontsize=14, fontweight='medium', color='#2c3e50')
        
        # Better x-axis labels
        ax.set_xticks(x)
        policy_labels = [policy.replace(' ', '\n') if len(policy) > 12 else policy for policy in policies]
        ax.set_xticklabels(policy_labels, fontsize=12, fontweight='medium', color='#34495e')
        
        # Set consistent y-axis limits with padding
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='y', labelsize=11, colors='#34495e')
        
        # Add value labels on bars with enhanced styling
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            
            # Position label above error bar
            label_y = height + std + 0.03
            
            # Style the label
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.3f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=thesis_colors[i % len(thesis_colors)], 
                           alpha=0.9, linewidth=1.5))
        
        # Add horizontal reference lines for common thresholds
        for threshold, color, style in [(0.5, '#e74c3c', '--'), (0.8, '#27ae60', ':')]:
            ax.axhline(y=threshold, color=color, linestyle=style, alpha=0.6, linewidth=1.5)
        
        # Add subtle border to subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.5)
    
    # Professional main title with subtitle
    fig.suptitle('ACT Policy Performance Analysis: Can Sorting Task', 
                fontsize=22, fontweight='bold', y=0.96, color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.93, 'Mean Success Rate ± Standard Deviation by Subtask', 
             ha='center', va='top', fontsize=14, style='italic', color='#7f8c8d')
    
    # Add legend for reference lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linestyle='--', alpha=0.6, label='50% Success'),
        Line2D([0], [0], color='#27ae60', linestyle=':', alpha=0.6, label='80% Success')
    ]
    
    # Position legend in the bottom right
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02),
              frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Professional layout with proper spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    
    # Save in multiple formats for thesis use
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.svg', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    
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


def create_statistical_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualizations for statistical analysis results."""
    
    # Perform statistical analysis to get results
    stats_results = perform_statistical_analysis(df, output_dir)
    
    # Create a figure with multiple subplots for statistical visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. P-value heatmap for policy differences
    ax1 = plt.subplot(2, 3, 1)
    
    # Extract p-values for policy ANOVA
    policy_anova = stats_results.get('policy_anova', {})
    tasks = list(policy_anova.keys())
    p_values = [policy_anova[task]['p_value'] for task in tasks]
    
    if tasks and p_values:
        # Create a heatmap-style bar plot for p-values
        colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'lightgray' for p in p_values]
        bars = ax1.barh(range(len(tasks)), [-np.log10(p) for p in p_values], color=colors)
        
        # Add significance threshold lines
        ax1.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax1.axvline(-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
        ax1.axvline(-np.log10(0.001), color='darkred', linestyle='--', alpha=0.7, label='p=0.001')
        
        ax1.set_yticks(range(len(tasks)))
        ax1.set_yticklabels([task.replace(' to ', '\nto ') for task in tasks], fontsize=10)
        ax1.set_xlabel('-log10(p-value)', fontsize=12)
        ax1.set_title('Policy Differences\n(ANOVA p-values)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add p-value labels on bars
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'p={p_val:.4f}', ha='left', va='center', fontsize=9)
    
    # 2. Effect sizes heatmap
    ax2 = plt.subplot(2, 3, 2)
    
    effect_sizes = stats_results.get('effect_sizes', {})
    if effect_sizes:
        # Create matrix for effect sizes
        policies = df['Policy'].unique()
        n_policies = len(policies)
        effect_matrix = np.zeros((len(tasks), n_policies * (n_policies - 1) // 2))
        comparison_labels = []
        
        col_idx = 0
        for i, policy1 in enumerate(policies):
            for policy2 in policies[i+1:]:
                comparison_labels.append(f"{policy1}\nvs\n{policy2}")
                for row_idx, task in enumerate(tasks):
                    comparison_key = f"{policy1} vs {policy2}"
                    if task in effect_sizes and comparison_key in effect_sizes[task]:
                        effect_matrix[row_idx, col_idx] = effect_sizes[task][comparison_key]['cohens_d']
                col_idx += 1
        
        if comparison_labels:
            im = ax2.imshow(effect_matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
            ax2.set_xticks(range(len(comparison_labels)))
            ax2.set_xticklabels(comparison_labels, rotation=45, ha='right', fontsize=9)
            ax2.set_yticks(range(len(tasks)))
            ax2.set_yticklabels([task.replace(' to ', '\nto ') for task in tasks], fontsize=10)
            ax2.set_title("Effect Sizes (Cohen's d)\nPolicy Comparisons", fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label("Cohen's d", fontsize=11)
            
            # Add text annotations
            for i in range(len(tasks)):
                for j in range(len(comparison_labels)):
                    value = effect_matrix[i, j]
                    if abs(value) > 0.1:  # Only show non-negligible effects
                        color = 'white' if abs(value) > 1 else 'black'
                        ax2.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                color=color, fontsize=8, fontweight='bold')
    
    # 3. Color effect analysis
    ax3 = plt.subplot(2, 3, 3)
    
    color_results = stats_results.get('color_analysis', {})
    if color_results:
        color_tasks = list(color_results.keys())
        color_p_values = [color_results[task]['p_value'] for task in color_tasks]
        
        colors_plot = ['green' if p < 0.001 else 'lightgreen' if p < 0.01 else 'yellow' if p < 0.05 else 'lightcoral' for p in color_p_values]
        bars = ax3.barh(range(len(color_tasks)), [-np.log10(p) for p in color_p_values], color=colors_plot)
        
        ax3.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        ax3.set_yticks(range(len(color_tasks)))
        ax3.set_yticklabels([task.replace(' to ', '\nto ') for task in color_tasks], fontsize=10)
        ax3.set_xlabel('-log10(p-value)', fontsize=12)
        ax3.set_title('Color Effects\n(ANOVA p-values)', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add p-value labels
        for i, (bar, p_val) in enumerate(zip(bars, color_p_values)):
            width = bar.get_width()
            ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'p={p_val:.4f}', ha='left', va='center', fontsize=9)
    
    # 4. Distribution comparison (violin plot)
    ax4 = plt.subplot(2, 3, 4)
    
    policies = df['Policy'].unique()
    policy_data = [df[df['Policy'] == policy]['Score'].values for policy in policies]
    
    # Create violin plot
    parts = ax4.violinplot(policy_data, positions=range(len(policies)), widths=0.7, showmeans=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax4.set_xticks(range(len(policies)))
    ax4.set_xticklabels(policies, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Success Rate', fontsize=12)
    ax4.set_title('Score Distributions\nby Policy', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Add statistical annotations
    from scipy.stats import kruskal
    if len(policy_data) > 1:
        try:
            h_stat, kruskal_p = kruskal(*policy_data)
            ax4.text(0.02, 0.98, f'Kruskal-Wallis\nH={h_stat:.2f}, p={kruskal_p:.4f}', 
                    transform=ax4.transAxes, va='top', ha='left', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        except:
            pass
    
    # 5. Color performance by policy
    ax5 = plt.subplot(2, 3, 5)
    
    colors_available = [c for c in df['Color'].unique() if c not in ['unknown', None]]
    if len(colors_available) >= 2:
        # Create grouped bar plot for color effects
        policy_color_means = df.groupby(['Policy', 'Color'])['Score'].mean().unstack(fill_value=0)
        
        x = np.arange(len(policies))
        width = 0.35
        
        if len(colors_available) == 2:
            color1, color2 = colors_available[:2]
            means1 = [policy_color_means.loc[policy, color1] if color1 in policy_color_means.columns and policy in policy_color_means.index else 0 for policy in policies]
            means2 = [policy_color_means.loc[policy, color2] if color2 in policy_color_means.columns and policy in policy_color_means.index else 0 for policy in policies]
            
            ax5.bar(x - width/2, means1, width, label=color1.title(), alpha=0.8)
            ax5.bar(x + width/2, means2, width, label=color2.title(), alpha=0.8)
        else:
            # More than 2 colors - use different approach
            for i, color in enumerate(colors_available[:4]):  # Limit to 4 colors for readability
                means = [policy_color_means.loc[policy, color] if color in policy_color_means.columns and policy in policy_color_means.index else 0 for policy in policies]
                offset = (i - len(colors_available)/2) * width/len(colors_available)
                ax5.bar(x + offset, means, width/len(colors_available), label=color.title(), alpha=0.8)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(policies, rotation=45, ha='right', fontsize=10)
        ax5.set_ylabel('Mean Success Rate', fontsize=12)
        ax5.set_title('Performance by\nCan Color', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim(0, 1)
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for policy in policies:
        policy_scores = df[df['Policy'] == policy]['Score']
        summary_data.append([
            policy,
            f"{policy_scores.mean():.3f}",
            f"{policy_scores.std():.3f}",
            f"{policy_scores.count()}"
        ])
    
    table = ax6.table(cellText=summary_data,
                      colLabels=['Policy', 'Mean', 'Std', 'N'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0.3, 1, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(policies) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax6.set_title('Summary Statistics\nby Policy', fontsize=14, fontweight='bold', y=0.95)
    
    # Add overall title and adjust layout
    fig.suptitle('Statistical Analysis of ACT Policy Performance on Can Sorting Task', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_dir / 'statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Statistical analysis plots saved to: {output_dir / 'statistical_analysis_plots.png'}")


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
        choices=["radar", "bars", "summary", "statistics", "stat_plots", "all"],
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
        selected_plots = ["radar", "bars", "summary", "statistics", "stat_plots"]
    
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
    
    if "stat_plots" in selected_plots:
        print("Creating statistical visualization plots...")
        create_statistical_plots(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
