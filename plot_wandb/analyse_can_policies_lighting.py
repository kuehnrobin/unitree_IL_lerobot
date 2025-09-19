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
import json
import warnings
from pathlib import Path
from typing import Tuple
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
import warnings


def format_policy_name(policy_name):
    """
    Format policy name to display subscripts correctly using matplotlib formatting.
    Converts underscore notation so only the next letter after underscore becomes subscript.
    Examples: 'S_LWA' -> 'S$_L$WA', 'R-S_LWA' -> 'R-S$_L$WA', 'A_B_C' -> 'A$_B$$_C$'
    """
    result = ""
    i = 0
    
    while i < len(policy_name):
        if policy_name[i] == '_' and i + 1 < len(policy_name):
            # Found underscore with character after it
            subscript_char = policy_name[i + 1]
            result += f"$_{{{subscript_char}}}$"
            i += 2  # Skip both underscore and the subscript character
        else:
            # Regular character, add it to result
            result += policy_name[i]
            i += 1
    
    return result


def parse_csv_data(csv_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Parse the complex CSV format with multiple policies and trials.
    
    Args:
        csv_path: Path to the lighting_test.csv file
        
    Returns:
        Tuple of:
        - Cleaned DataFrame with columns: Policy, Trial, Color, Task, Score, Time
        - Time info dictionary with min_time, max_time, time_range in seconds
    """
    # Read raw CSV
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)
    
    # Define the subtasks we're tracking (from the CSV comments)
    subtasks = [
        "Hand Move to Can",
        "Hand Grasp Can", 
        "Hand Move to Correct Box",
        "Can in Correct Box"
    ]
    
    # Initialize list to store parsed data
    parsed_data = []
    
    # First pass: collect all execution times for relative normalization
    all_execution_times = []
    
    # Find policy start rows - look for policy names in the first column
    policy_start_rows = []
    for idx, row in raw_df.iterrows():
        if pd.notna(row[0]) and str(row[0]).strip() != '':
            # Skip comment/explanation rows
            cell_value = str(row[0]).strip()
            if (cell_value not in ['Config ID', 'Config iD', 'Row dose not exist in Data (Just for Comments)', 
                                  'Policy Name', 'Subtask 1', 'Subtask 2', 'Subtask 3', 'Subtask 4', 'Subtask 5'] 
                and not cell_value.startswith('Trail number')):
                policy_start_rows.append(idx)
    
    print(f"Found policies at rows: {policy_start_rows}")
    
    # First pass: collect all execution times for relative normalization
    for policy_idx, start_row in enumerate(policy_start_rows):
        # Process Return to Home Position row to extract execution times
        home_pos_row_idx = start_row + len(subtasks) + 1  # +1 for the color row
        if home_pos_row_idx < len(raw_df):
            home_pos_row = raw_df.iloc[home_pos_row_idx, 1:]  # Skip first column (task name)
            
            # Process 5 trials, each with 8 columns
            for trial_idx in range(5):
                col_start = trial_idx * 8
                time_col_idx = col_start + 7  # Time is in the 8th column of each trial
                
                if time_col_idx < len(home_pos_row):
                    time_val = home_pos_row.iloc[time_col_idx]
                    if pd.notna(time_val) and str(time_val).strip() not in ['None', 'end', 'time', '']:
                        time_str = str(time_val).strip()
                        # Parse time format like "04:03" to seconds
                        if ':' in time_str:
                            try:
                                time_parts = time_str.split(':')
                                time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                all_execution_times.append(time_seconds)
                            except (ValueError, IndexError):
                                pass
    
    # Calculate min and max execution times for relative normalization
    if all_execution_times:
        min_time = min(all_execution_times)
        max_time = max(all_execution_times)
        time_range = max_time - min_time if max_time > min_time else 1
    else:
        min_time, max_time, time_range = 180, 350, 170  # Default values based on visible data
    
    print(f"Time range: {min_time/60:.1f} - {max_time/60:.1f} minutes")
    
    # Second pass: actual data processing
    for policy_idx, start_row in enumerate(policy_start_rows):
        # Extract policy name
        policy_name = str(raw_df.iloc[start_row, 0]).strip()
        
        print(f"Processing policy: {policy_name}")
        
        # Extract color sequence (row after policy name)
        color_row_idx = start_row + 1
        if color_row_idx >= len(raw_df):
            continue
            
        color_row = raw_df.iloc[color_row_idx, 1:]  # Skip first column
        colors = []
        
        # Parse colors for all 5 trials (8 columns each)
        for trial_idx in range(5):
            col_start = trial_idx * 8
            trial_colors = []
            
            # Get colors for first 6 columns of this trial
            for col_offset in range(6):
                col_idx = col_start + col_offset
                if col_idx < len(color_row):
                    color_val = color_row.iloc[col_idx]
                    if pd.notna(color_val):
                        color_str = str(color_val).strip().lower()
                        if color_str in ['red', 'green']:
                            trial_colors.append(color_str)
            
            # Determine dominant color for this trial
            if trial_colors:
                dominant_color = max(set(trial_colors), key=trial_colors.count)
                colors.append(dominant_color)
            else:
                colors.append('unknown')
        
        print(f"Trial colors: {colors}")
        
        # Process each subtask
        for task_offset, task_name in enumerate(subtasks):
            task_row_idx = start_row + 2 + task_offset  # +2 to skip policy name and color row
            if task_row_idx >= len(raw_df):
                continue
                
            task_row = raw_df.iloc[task_row_idx, 1:]  # Skip task name column
            
            # Process each trial
            for trial_idx in range(5):
                col_start = trial_idx * 8
                
                # Collect scores for first 6 columns of this trial
                trial_scores = []
                
                for col_offset in range(6):
                    col_idx = col_start + col_offset
                    if col_idx < len(task_row):
                        score = task_row.iloc[col_idx]
                        if pd.notna(score) and str(score).strip() not in ['None', 'end', 'time', '']:
                            try:
                                score_val = float(score)
                                trial_scores.append(score_val)
                            except ValueError:
                                pass
                
                # Calculate mean score for this trial and task
                if trial_scores:
                    mean_score = np.mean(trial_scores)
                    trial_color = colors[trial_idx] if trial_idx < len(colors) else 'unknown'
                    
                    parsed_data.append({
                        'Policy': policy_name,
                        'Trial': trial_idx + 1,
                        'Color': trial_color,
                        'Task': task_name,
                        'Score': mean_score
                    })
        
        # Process Return to Home Position data
        home_pos_row_idx = start_row + 2 + len(subtasks)  # After all subtask rows
        if home_pos_row_idx < len(raw_df):
            home_pos_row = raw_df.iloc[home_pos_row_idx, 1:]  # Skip task name column
            
            for trial_idx in range(5):
                col_start = trial_idx * 8
                
                # Extract end position score (7th column of each trial)
                end_pos_score = None
                time_score = None
                
                end_pos_col_idx = col_start + 6  # End position (7th column)
                time_col_idx = col_start + 7     # Time (8th column)
                
                if end_pos_col_idx < len(home_pos_row):
                    end_pos_val = home_pos_row.iloc[end_pos_col_idx]
                    if pd.notna(end_pos_val) and str(end_pos_val).strip() not in ['None', 'end', 'time', '']:
                        try:
                            end_pos_score = float(end_pos_val)
                        except ValueError:
                            end_pos_score = 1.0 if str(end_pos_val).strip() == '1' else 0.0
                
                if time_col_idx < len(home_pos_row):
                    time_val = home_pos_row.iloc[time_col_idx]
                    if pd.notna(time_val) and str(time_val).strip() not in ['None', 'end', 'time', '']:
                        time_str = str(time_val).strip()
                        if ':' in time_str:
                            try:
                                time_parts = time_str.split(':')
                                time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                # Convert to relative normalized score (faster = higher score)
                                time_score = (max_time - time_seconds) / time_range if time_range > 0 else 0.0
                            except (ValueError, IndexError):
                                pass
                
                trial_color = colors[trial_idx] if trial_idx < len(colors) else 'unknown'
                
                # Add Return to Home Position data
                if end_pos_score is not None:
                    parsed_data.append({
                        'Policy': policy_name,
                        'Trial': trial_idx + 1,
                        'Color': trial_color,
                        'Task': 'Return to Home Position',
                        'Score': end_pos_score
                    })
                
                # Add Execution Time data
                if time_score is not None:
                    parsed_data.append({
                        'Policy': policy_name,
                        'Trial': trial_idx + 1,
                        'Color': trial_color,
                        'Task': 'Execution Time',
                        'Score': time_score
                    })
    
    # Create DataFrame from parsed data
    df = pd.DataFrame(parsed_data)
    
    print(f"Parsed {len(df)} data points")
    if not df.empty:
        print(f"Policies: {df['Policy'].unique()}")
        print(f"Tasks: {df['Task'].unique()}")
        print(f"Colors: {df['Color'].unique()}")
    
    # Store time information for axis labeling
    time_info = {
        'min_time_seconds': min_time,
        'max_time_seconds': max_time,
        'time_range_seconds': time_range,
        'min_time_minutes': min_time / 60.0,
        'max_time_minutes': max_time / 60.0
    }
    
    # Calculate policy-specific execution times for labeling
    policy_times = {}
    if not df.empty:
        for policy in df['Policy'].unique():
            policy_time_data = df[(df['Policy'] == policy) & (df['Task'] == 'Execution Time')]
            if not policy_time_data.empty:
                # Convert normalized score back to actual time
                mean_score = policy_time_data['Score'].mean()
                # Inverse of normalization: time = max_time - (score * time_range)
                actual_time_seconds = max_time - (mean_score * time_range)
                policy_times[policy] = {
                    'seconds': actual_time_seconds,
                    'minutes': actual_time_seconds / 60.0
                }
    
    time_info['policy_times'] = policy_times
    
    # Calculate total policy scores with optional weighting
    if not df.empty:
        # Define task weights (all set to 1.0 for unweighted average)
        task_weights = {
            "Hand Move to Can": 1.0,
            "Hand Grasp Can": 1.0,
            "Hand Move to Correct Box": 1.0,
            "Can in Correct Box": 1.0,
            "Return to Home Position": 1.0,
            "Execution Time": 1.0
        }
        
        # Calculate weighted total scores for each policy and trial
        for policy in df['Policy'].unique():
            for trial in df[df['Policy'] == policy]['Trial'].unique():
                policy_trial_data = df[(df['Policy'] == policy) & (df['Trial'] == trial)]
                
                if len(policy_trial_data) > 0:
                    # Calculate weighted average score
                    total_score = 0
                    total_weight = 0
                    trial_color = policy_trial_data['Color'].iloc[0]
                    
                    for task in task_weights.keys():
                        task_data = policy_trial_data[policy_trial_data['Task'] == task]
                        if len(task_data) > 0:
                            task_score = task_data['Score'].iloc[0]
                            weight = task_weights[task]
                            total_score += task_score * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_average = total_score / total_weight
                        
                        # Add total score as a new task
                        df = pd.concat([df, pd.DataFrame([{
                            'Policy': policy,
                            'Trial': trial,
                            'Color': trial_color,
                            'Task': 'Total Score',
                            'Score': weighted_average
                        }])], ignore_index=True)
    
    return df, time_info


def create_radar_chart(df: pd.DataFrame, time_info: dict, output_dir: Path, include_total_score: bool = True) -> None:
    """Create a radar chart comparing all policies across subtasks including End Position and Time."""
    
    # Calculate mean scores per policy and task
    policy_stats = df.groupby(['Policy', 'Task'])['Score'].mean().unstack(fill_value=0)
    
    # Ensure all subtasks are present including new metrics
    subtasks = [
        "Hand Move to Can",
        "Hand Grasp Can", 
        "Hand Move to Correct Box",
        "Can in Correct Box",
        "Return to Home Position",
        "Execution Time"
    ]
    
    if include_total_score:
        subtasks.append("Total Score")
    for task in subtasks:
        if task not in policy_stats.columns:
            policy_stats[task] = 0
    
    policy_stats = policy_stats[subtasks]  # Reorder columns
    
    # Professional color scheme
    thesis_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Set up radar chart with better proportions
    N = len(subtasks)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Enlarge figure for more space around chart
    fig, ax = plt.subplots(figsize=(15, 14), subplot_kw=dict(projection='polar'), dpi=150)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Enhanced grid styling
    ax.grid(True, alpha=0.4, linewidth=0.8, color='gray')
    ax.set_facecolor('#fafafa')
    
    # Plot each policy with enhanced styling
    n_policies = len(policy_stats)
    for idx, (policy, scores) in enumerate(policy_stats.iterrows()):
        values = scores.tolist()
        values += values[:1]  # Complete the circle
        color = thesis_colors[idx % len(thesis_colors)]

        ax.plot(angles, values, 'o-', linewidth=3, label=format_policy_name(policy), color=color,
               markersize=8, markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=2, alpha=0.9)

        ax.fill(angles, values, alpha=0.08, color=color)

        # Enhanced dynamic label positioning: spread labels by policy index
        # so multiple policies at the same angle don't overlap.
        for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            if value > 0.05:
                angle_deg = (angle * 180 / pi) % 360
                # Normalized policy offset in [-1, 1]
                norm_idx = (idx - (n_policies - 1) / 2) / ((n_policies - 1) / 2) if n_policies > 1 else 0.0

                # Stronger angle jitter at top/bottom to spread horizontally more
                if angle_deg <= 45 or (135 < angle_deg <= 225) or angle_deg >= 315:
                    angle_jitter = 0.34  # increased spread for top/bottom spokes
                else:
                    angle_jitter = 0.22  # left/right
                angle_shifted = angle + norm_idx * angle_jitter

                # Place numeric labels outside the data region but inside figure
                base_tb = 0.12 if value < 0.3 else 0.10
                base_lr = 0.10
                signed_radial = 0.03 * norm_idx
                outside_min_top = 1.10
                outside_min_lr  = 1.12
                outside_min_bot = 1.12
                outside_max = 1.18  # keep below tick label at left that we move to ~1.24

                if angle_deg <= 45 or angle_deg >= 315:
                    label_r = max(value + base_tb + abs(signed_radial), outside_min_top)
                    label_r = min(label_r, outside_max)
                    ha, va = 'center', 'bottom'
                elif 45 < angle_deg <= 135:
                    label_r = max(value + base_lr + abs(signed_radial), outside_min_lr)
                    label_r = min(label_r, outside_max)
                    ha, va = 'left', 'center'
                elif 135 < angle_deg <= 225:
                    label_r = max(value + base_tb + abs(signed_radial), outside_min_bot)
                    label_r = min(label_r, outside_max)
                    ha, va = 'center', 'top'
                else:
                    label_r = max(value + base_lr + abs(signed_radial), outside_min_lr)
                    label_r = min(label_r, outside_max)
                    ha, va = 'right', 'center'

                # Determine label text based on task type
                task_name = subtasks[j] if j < len(subtasks) else "Unknown"
                if task_name == "Execution Time" and policy in time_info.get('policy_times', {}):
                    # Show actual time in minutes for execution time
                    actual_minutes = time_info['policy_times'][policy]['minutes']
                    label_text = f'{actual_minutes:.1f}min'
                else:
                    # Show normalized score for other tasks
                    label_text = f'{value:.2f}'

                ax.text(angle_shifted, label_r, label_text,
                        ha=ha, va=va, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                                  edgecolor=color, alpha=0.85, linewidth=1.2),
                        zorder=10, clip_on=False)

    # Enhanced axis customization
    ax.set_xticks(angles[:-1])
    task_labels = [
        "Move to\nCan",
        "Grasp\nCan", 
        "Move to\nCorrect Box",
        "Place Can in\nCorrect Box",
        "Return to\nHome Position",
        f"Execution\nTime\n({time_info['min_time_minutes']:.1f}-\n{time_info['max_time_minutes']:.1f} min)"
    ]
    
    if include_total_score:
        task_labels.append("Total\nScore")
    ax.set_xticklabels(task_labels, fontsize=12, fontweight='bold', ha='center')
    ax.tick_params(axis='x', pad=32)  # push all task labels outward

    # Increase radial limit to make room for outside labels
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11, alpha=0.8, fontweight='medium')

    # Add radial grid lines at specific values
    for tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot([0, 2*pi], [tick, tick], color='gray', alpha=0.3, linewidth=0.8)
    
    # Legend stays outside bottom-right
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.00, -0.06),
                      borderaxespad=0.0, frameon=True, fancybox=True, shadow=True,
                      fontsize=10, title='ACT Policies', title_fontsize=11)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')

    # Titles: bring closer to the figure at left
    fig.suptitle('Policy Performance Comparison on Can Sorting Task',
                 x=0.26, y=1.0, size=18, fontweight='bold', color='#2c3e50', ha='left')
    fig.text(0.33, 0.98, 'Success Rate by Subtask (0.0 = Failure, 1.0 = Success)', 
             ha='left', va='top', fontsize=12, style='italic', color='#6c757d')

    #plt.tight_layout(pad=2, rect=[0.15, 0.00, 0.83, 0.88])
    plt.tight_layout(rect=[0.00, 0.00, 1.00, 0.8])  # Changed from default to reserve top space
    plt.tight_layout()
    # Save with multiple formats for thesis use
    plt.savefig(output_dir / 'radar_chart_policy_comparison.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'radar_chart_policy_comparison.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    #plt.savefig(output_dir / 'radar_chart_policy_comparison.svg', 
    #            bbox_inches='tight', facecolor='white', edgecolor='none')
    
    #plt.show()


def create_grouped_bar_plot(df: pd.DataFrame, time_info: dict, output_dir: Path) -> None:
    """Create beautiful grouped bar plots with error bars for each subtask including End Position and Time."""
    
    # Calculate statistics
    stats = df.groupby(['Policy', 'Task'])['Score'].agg(['mean', 'std', 'count']).reset_index()
    
    subtasks = [
        "Hand Move to Can", 
        "Hand Grasp Can", 
        "Hand Move to Correct Box", 
        "Can in Correct Box",
        "Return to Home Position",
        "Execution Time"
    ]
    policies = stats['Policy'].unique()
    
    # Professional color scheme for thesis
    thesis_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Set up the plot with better spacing and professional styling - now 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(18, 20), dpi=150)
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
        bars = ax.bar(x, means, yerr=stds, capsize=8,
                     color=[thesis_colors[i % len(thesis_colors)] for i in range(len(policies))],
                     alpha=0.85, edgecolor='white', linewidth=2,
                     error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50', 'alpha': 0.8})
        
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
        policy_labels = [format_policy_name(policy).replace(' ', '\n') if len(policy) > 12 else format_policy_name(policy) for policy in policies]
        ax.set_xticklabels(policy_labels, fontsize=12, fontweight='medium', color='#34495e', rotation=45, ha='right')
        
        # Set consistent y-axis limits with padding to accommodate error bars and labels
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.tick_params(axis='y', labelsize=11, colors='#34495e')
        
        # Add a horizontal line indicating 100% success rate
        ax.axhline(y=1.0, color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5, label='100% Success')
        
        # Add value labels on bars with enhanced styling
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            
            # Position label above error bar
            label_y = height + std + 0.03
            
            # Determine label text based on task type
            policy = policies[i]
            if task == "Execution Time" and policy in time_info.get('policy_times', {}):
                # Show actual time in minutes for execution time
                actual_minutes = time_info['policy_times'][policy]['minutes']
                label_text = f'{actual_minutes:.1f}min'
            else:
                # Show normalized score for other tasks
                label_text = f'{mean:.3f}'
            
            # Style the label
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   label_text, ha='center', va='bottom', 
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
    fig.text(0.5, 0.92, 'Mean Success Rate ± Standard Deviation by Subtask', 
             ha='center', va='top', fontsize=14, style='italic', color='#7f8c8d')
    
    # Add legend for reference lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5, label='100% Success'),
        Line2D([0], [0], color='#e74c3c', linestyle='--', alpha=0.6, label='50% Success'),
        Line2D([0], [0], color='#27ae60', linestyle=':', alpha=0.6, label='80% Success')
    ]
    
    # Position legend in the bottom right
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02),
              frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Professional layout with proper spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    
    # Save in multiple formats for thesis use
    #plt.savefig(output_dir / 'grouped_bar_plot_with_errors.png', 
    #            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    #plt.savefig(output_dir / 'grouped_bar_plot_with_errors.svg', 
    #            bbox_inches='tight', facecolor='white', edgecolor='none')
    
    #plt.show()


def create_total_score_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create focused analysis for Total Score performance."""
    
    # Filter for Total Score data
    total_score_data = df[df['Task'] == 'Total Score']
    
    if total_score_data.empty:
        print("No Total Score data found")
        return
    
    # Professional styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 1. Bar plot of total scores
    policy_stats = total_score_data.groupby('Policy')['Score'].agg(['mean', 'std', 'count'])
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    x = np.arange(len(policy_stats.index))
    
    bars = ax1.bar(x, policy_stats['mean'], yerr=policy_stats['std'], 
                   capsize=8, color=colors[:len(policy_stats)], alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_policy_name(policy) for policy in policy_stats.index], rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('Total Score', fontsize=14, fontweight='bold')
    ax1.set_title('Total Policy Performance Score', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, policy_stats['mean'], policy_stats['std']):
        height = bar.get_height()
        # Move labels higher to avoid overlap with error bars
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.04,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Color-based performance for total score
    if len(total_score_data['Color'].unique()) > 1:
        color_policy_stats = total_score_data.groupby(['Policy', 'Color'])['Score'].mean().unstack(fill_value=0)
        
        if 'red' in color_policy_stats.columns and 'green' in color_policy_stats.columns:
            # Add averaged totals row
            policies_list = list(color_policy_stats.index)
            
            # Calculate average across all policies for each color
            red_avg = color_policy_stats['red'].mean()
            green_avg = color_policy_stats['green'].mean()
            
            # Create extended data including the average
            extended_policies = policies_list + ['Average']
            red_values = list(color_policy_stats['red']) + [red_avg]
            green_values = list(color_policy_stats['green']) + [green_avg]
            
            x2 = np.arange(len(extended_policies))
            width = 0.35
            
            bars1 = ax2.bar(x2 - width/2, red_values, width, 
                           label='Red Cans', color='#e74c3c', alpha=0.8)
            bars2 = ax2.bar(x2 + width/2, green_values, width,
                           label='Green Cans', color='#2ecc71', alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars1, red_values)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            for i, (bar, value) in enumerate(zip(bars2, green_values)):
                height = bar.get_height()
                # Move green labels back to original position
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.set_xticks(x2)
            ax2.set_xticklabels([format_policy_name(policy) for policy in extended_policies], rotation=45, ha='right', fontsize=12)
            
            # Add a visual separator before the sum column
            ax2.axvline(x=len(policies_list) - 0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.set_ylabel('Total Score', fontsize=14, fontweight='bold')
        ax2.set_title('Total Score by Can Color', fontsize=16, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'total_score_analysis.pdf', bbox_inches='tight')
    plt.close()


def create_end_position_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create focused analysis for End Position performance."""
    
    # Filter for End Position data
    end_pos_data = df[df['Task'] == 'Return to Home Position']
    
    if end_pos_data.empty:
        print("No End Position data found")
        return
    
    # Professional styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 1. Bar plot of end position success rates
    policy_stats = end_pos_data.groupby('Policy')['Score'].agg(['mean', 'std', 'count'])
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    x = np.arange(len(policy_stats.index))
    
    bars = ax1.bar(x, policy_stats['mean'], yerr=policy_stats['std'], 
                   capsize=8, color=colors[:len(policy_stats)], alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_policy_name(policy) for policy in policy_stats.index], rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
    ax1.set_title('Return to Home Position Success Rate', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, policy_stats['mean'], policy_stats['std']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Color-based performance for end position
    if len(end_pos_data['Color'].unique()) > 1:
        color_policy_stats = end_pos_data.groupby(['Policy', 'Color'])['Score'].mean().unstack(fill_value=0)
        
        x2 = np.arange(len(color_policy_stats.index))
        width = 0.35
        
        if 'red' in color_policy_stats.columns and 'green' in color_policy_stats.columns:
            bars1 = ax2.bar(x2 - width/2, color_policy_stats['red'], width, 
                           label='Red Cans', color='#e74c3c', alpha=0.8)
            bars2 = ax2.bar(x2 + width/2, color_policy_stats['green'], width,
                           label='Green Cans', color='#2ecc71', alpha=0.8)
        
        ax2.set_xticks(x2)
        ax2.set_xticklabels([format_policy_name(policy) for policy in color_policy_stats.index], rotation=45, ha='right', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=14, fontweight='bold')
        ax2.set_title('End Position by Can Color', fontsize=16, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'end_position_analysis.pdf', bbox_inches='tight')
    plt.close()


def create_time_analysis(df: pd.DataFrame, time_info: dict, output_dir: Path) -> None:
    """Create focused analysis for execution time performance."""
    
    # Filter for Time data
    time_data = df[df['Task'] == 'Execution Time']
    
    if time_data.empty:
        print("No Execution Time data found")
        return
    
    # Professional styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 1. Bar plot of time efficiency (higher score = faster execution)
    policy_stats = time_data.groupby('Policy')['Score'].agg(['mean', 'std', 'count'])
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    x = np.arange(len(policy_stats.index))
    
    bars = ax1.bar(x, policy_stats['mean'], yerr=policy_stats['std'], 
                   capsize=8, color=colors[:len(policy_stats)], alpha=0.85,
                   edgecolor='white', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([format_policy_name(policy) for policy in policy_stats.index], rotation=45, ha='right', fontsize=12)
    ax1.set_ylabel('Time Efficiency Score', fontsize=14, fontweight='bold')
    ax1.set_title(f'Execution Time Efficiency\n(Range: {time_info["min_time_minutes"]:.1f}-{time_info["max_time_minutes"]:.1f} minutes)', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add secondary y-axis showing actual minutes
    ax1_twin = ax1.twinx()
    
    # Convert efficiency scores back to actual minutes for secondary axis
    def score_to_minutes(score):
        # Reverse the normalization: score = (max_time - time_seconds) / time_range
        # Therefore: time_seconds = max_time - (score * time_range)
        time_seconds = time_info['max_time_seconds'] - (score * time_info['time_range_seconds'])
        return time_seconds / 60.0
    
    # Set up secondary axis ticks
    efficiency_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    minute_ticks = [score_to_minutes(score) for score in efficiency_ticks]
    ax1_twin.set_ylim(0, 1.1)
    ax1_twin.set_yticks(efficiency_ticks)
    ax1_twin.set_yticklabels([f'{min_val:.1f}' for min_val in minute_ticks], fontsize=11)
    ax1_twin.set_ylabel('Execution Time (minutes)', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mean, std in zip(bars, policy_stats['mean'], policy_stats['std']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Violin plot showing distribution of time scores
    policies = time_data['Policy'].unique()
    time_distributions = [time_data[time_data['Policy'] == policy]['Score'].values for policy in policies]
    
    parts = ax2.violinplot(time_distributions, positions=range(len(policies)), widths=0.7, showmeans=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    ax2.set_xticks(range(len(policies)))
    ax2.set_xticklabels([format_policy_name(policy) for policy in policies], rotation=45, ha='right', fontsize=12)
    ax2.set_ylabel('Time Efficiency Score', fontsize=14, fontweight='bold')
    ax2.set_title(f'Time Efficiency Distribution\n(Higher score = faster execution)', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add secondary y-axis for violin plot too
    ax2_twin = ax2.twinx()
    efficiency_ticks_violin = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    minute_ticks_violin = [score_to_minutes(score) for score in efficiency_ticks_violin]
    ax2_twin.set_ylim(0, 1)
    ax2_twin.set_yticks(efficiency_ticks_violin)
    ax2_twin.set_yticklabels([f'{min_val:.1f}' for min_val in minute_ticks_violin], fontsize=11)
    ax2_twin.set_ylabel('Execution Time (minutes)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_analysis.pdf', bbox_inches='tight')
    plt.close()


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
        
        # Create policy number mapping
        policy_numbers = {policy: i+1 for i, policy in enumerate(policies)}
        
        effect_matrix = np.zeros((len(tasks), n_policies * (n_policies - 1) // 2))
        comparison_labels = []
        
        col_idx = 0
        for i, policy1 in enumerate(policies):
            for policy2 in policies[i+1:]:
                # Use numbers instead of policy names
                comparison_labels.append(f"[{policy_numbers[policy1]}] vs [{policy_numbers[policy2]}]")
                for row_idx, task in enumerate(tasks):
                    comparison_key = f"{policy1} vs {policy2}"
                    if task in effect_sizes and comparison_key in effect_sizes[task]:
                        effect_matrix[row_idx, col_idx] = effect_sizes[task][comparison_key]['cohens_d']
                col_idx += 1
        
        if comparison_labels:
            im = ax2.imshow(effect_matrix, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
            ax2.set_xticks(range(len(comparison_labels)))
            ax2.set_xticklabels(comparison_labels, rotation=45, ha='center', fontsize=9)  # Changed rotation to 0
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
    ax4.set_xticklabels([format_policy_name(policy) for policy in policies], rotation=45, ha='right', fontsize=10)
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
        ax5.set_xticklabels([format_policy_name(policy) for policy in policies], rotation=45, ha='right', fontsize=10)
        ax5.set_ylabel('Mean Success Rate', fontsize=12)
        ax5.set_title('Performance by\nCan Color', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim(0, 1)
    
    # 6. Summary statistics table and policy legend
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
                      loc='upper center',
                      bbox=[0, 0.5, 1, 0.4])  # Changed bbox to make room for legend below

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
    
    ax6.set_title('Summary Statistics\nby Policy', fontsize=14, fontweight='bold', y=1.0)

    # Add policy number legend below the table
    legend_y_start = 0.4
    ax6.text(0.5, legend_y_start, 'Policy Number Legend:', 
             ha='center', va='top', fontsize=12, fontweight='bold',
             transform=ax6.transAxes)

    # Create policy legend with numbers
    for i, policy in enumerate(policies):
        y_pos = legend_y_start - 0.05 - (i * 0.06)
        ax6.text(0.5, y_pos, f'[{i+1}] {policy}', 
                 ha='center', va='top', fontsize=10,
                 transform=ax6.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

    # Add overall title and adjust layout
    fig.suptitle('Statistical Analysis of ACT Policy Performance on Can Sorting Task', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    #plt.savefig(output_dir / 'statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'statistical_analysis_plots.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    #plt.show()
    
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
        default="plot_wandb/lighting_test.csv",
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
        choices=["radar", "bars", "summary", "statistics", "stat_plots", "end_position", "time_analysis", "total_score", "all"],
        default=["all"],
        help="Choose which plots to create"
    )
    
    parser.add_argument(
        "--include_total_score",
        action="store_true",
        default=True,
        help="Include total score in radar chart (default: True)"
    )
    
    parser.add_argument(
        "--no_total_score",
        action="store_false",
        dest="include_total_score",
        help="Exclude total score from radar chart"
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
        df, time_info = parse_csv_data(args.csv_path)
        print(f"Successfully parsed {len(df)} data points")
        print(f"Policies found: {df['Policy'].unique()}")
        print(f"Tasks found: {df['Task'].unique()}")
        print(f"Time range: {time_info['min_time_minutes']:.1f}-{time_info['max_time_minutes']:.1f} minutes")
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return 1
    
    # Save parsed data for inspection
    df.to_csv(output_dir / 'parsed_data.csv', index=False)
    print(f"Parsed data saved to: {output_dir / 'parsed_data.csv'}")
    
    # Determine which plots to create
    selected_plots = args.plots
    if "all" in selected_plots:
        selected_plots = ["radar", "bars", "summary", "statistics", "stat_plots", "end_position", "time_analysis", "total_score"]
    
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
        create_radar_chart(df, time_info, output_dir, args.include_total_score)
    
    if "bars" in selected_plots:
        print("Creating grouped bar plots...")
        create_grouped_bar_plot(df, time_info, output_dir)
    
    if "summary" in selected_plots:
        print("Generating summary statistics...")
        create_summary_statistics(df, output_dir)
    
    if "statistics" in selected_plots:
        print("Performing statistical analysis...")
        perform_statistical_analysis(df, output_dir)
    
    if "stat_plots" in selected_plots:
        print("Creating statistical visualization plots...")
        create_statistical_plots(df, output_dir)
    
    if "end_position" in selected_plots:
        print("Creating end position analysis...")
        create_end_position_analysis(df, output_dir)
    
    if "time_analysis" in selected_plots:
        print("Creating time analysis...")
        create_time_analysis(df, time_info, output_dir)
    
    if "total_score" in selected_plots:
        print("Creating total score analysis...")
        create_total_score_analysis(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
