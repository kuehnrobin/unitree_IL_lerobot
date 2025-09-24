#!/usr/bin/env python3
"""
Analysis and visualization of ACT policy performance on cube manipulation tasks.
This script reads results from cubes_policies.csv and creates comparative visualizations.
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
    Parse the complex CSV format with multiple policies and trials for cube manipulation.
    
    Args:
        csv_path: Path to the cubes_policies.csv file
        
    Returns:
        Tuple of:
        - Cleaned DataFrame with columns: Policy, Trial, Color, Hand, Task, Score, Time
        - Time info dictionary with min_time, max_time, time_range in seconds
    """
    # Read raw CSV
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)
    
    # Define the subtasks we're tracking for cube manipulation
    subtasks = [
        "Hand Move to Cube",
        "Hand Grasp Cube",
        "Hand Move to Box",
        "Cube in Box"
    ]
    
    # Initialize list to store parsed data
    parsed_data = []
    
    # First pass: collect all execution times for relative normalization
    all_execution_times = []
    
    # Process each policy section
    policy_start_rows = []
    for idx, row in raw_df.iterrows():
        if pd.notna(row[0]) and any(policy in str(row[0]).lower() for policy in ['cubes_r', 'r-', 'dino']):
            policy_start_rows.append(idx)
    
    # First pass: collect all execution times for relative normalization
    for policy_idx, start_row in enumerate(policy_start_rows):
        # Find the end of this policy section
        end_row = policy_start_rows[policy_idx + 1] if policy_idx + 1 < len(policy_start_rows) else len(raw_df)
        
        # Extract color sequence (first row after policy name)
        color_row = raw_df.iloc[start_row, 1:]
        colors = [str(c).lower() if pd.notna(c) and str(c).lower() in ['red', 'green', 'black'] else None 
                 for c in color_row]
        
        # Extract hand usage (second row after policy name)
        hand_row_idx = start_row + 1
        hand_row = raw_df.iloc[hand_row_idx, 1:] if hand_row_idx < end_row else []
        hands = [str(h).lower() if pd.notna(h) and str(h).lower() in ['left', 'right'] else None 
                for h in hand_row]
        
        # Count valid manipulations (where we have both color and hand data)
        valid_manipulations = min(len([c for c in colors if c is not None]), 
                                len([h for h in hands if h is not None]))
        
        # Process Execution Time data to extract times
        exec_time_row_idx = start_row + len(subtasks) + 2  # +2 for color and hand rows
        if exec_time_row_idx < end_row:
            exec_time_row = raw_df.iloc[exec_time_row_idx, 1:]
            
            for manip_idx in range(valid_manipulations):
                if manip_idx < len(exec_time_row):
                    time_val = exec_time_row.iloc[manip_idx]
                    if pd.notna(time_val) and str(time_val) not in ['None', 'end', 'time', '']:
                        time_str = str(time_val)
                        # Parse time format like "01:05" to seconds
                        if ':' in time_str:
                            try:
                                time_parts = time_str.split(':')
                                time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                all_execution_times.append(time_seconds)
                            except ValueError:
                                pass
    
    # Calculate min and max execution times for relative normalization
    if all_execution_times:
        min_time = min(all_execution_times)
        max_time = max(all_execution_times)
        time_range = max_time - min_time if max_time > min_time else 1  # Avoid division by zero
    else:
        min_time, max_time, time_range = 0, 900, 900  # Fallback to old method
    
    # Second pass: actual data processing with relative time normalization
    for policy_idx, start_row in enumerate(policy_start_rows):
        # Extract policy name - handle abbreviated names like R-13456-P
        policy_name = str(raw_df.iloc[start_row, 0]).replace('cubes_', '').upper()
        
        # Find the end of this policy section
        end_row = policy_start_rows[policy_idx + 1] if policy_idx + 1 < len(policy_start_rows) else len(raw_df)
        
        # Extract color sequence
        color_row = raw_df.iloc[start_row, 1:]
        colors = [str(c).lower() if pd.notna(c) and str(c).lower() in ['red', 'green', 'black'] else None 
                 for c in color_row]
        
        # Extract hand usage
        hand_row_idx = start_row + 1
        hand_row = raw_df.iloc[hand_row_idx, 1:] if hand_row_idx < end_row else []
        hands = [str(h).lower() if pd.notna(h) and str(h).lower() in ['left', 'right'] else None 
                for h in hand_row]
        
        # Count valid manipulations
        valid_manipulations = min(len([c for c in colors if c is not None]), 
                                len([h for h in hands if h is not None]))
        
        # Process each subtask
        for task_offset, task_name in enumerate(subtasks):
            task_row_idx = start_row + 2 + task_offset  # +2 for color and hand rows
            if task_row_idx >= end_row:
                continue
                
            task_row = raw_df.iloc[task_row_idx, 1:]
            
            # Extract scores for each manipulation
            for manip_idx in range(valid_manipulations):
                if manip_idx < len(task_row):
                    score = task_row.iloc[manip_idx]
                    if pd.notna(score) and str(score) not in ['None', 'end', 'time', '']:
                        try:
                            score_val = float(score)
                            
                            # Get manipulation color and hand
                            manip_color = colors[manip_idx] if manip_idx < len(colors) else 'unknown'
                            manip_hand = hands[manip_idx] if manip_idx < len(hands) else 'unknown'
                            
                            parsed_data.append({
                                'Policy': policy_name,
                                'Trial': manip_idx + 1,  # Each manipulation is a separate trial
                                'Color': manip_color,
                                'Hand': manip_hand,
                                'Task': task_name,
                                'Score': score_val
                            })
                        except ValueError:
                            pass
        
        # Process Execution Time data
        exec_time_row_idx = start_row + len(subtasks) + 2  # +2 for color and hand rows
        if exec_time_row_idx < end_row:
            exec_time_row = raw_df.iloc[exec_time_row_idx, 1:]
            
            for manip_idx in range(valid_manipulations):
                if manip_idx < len(exec_time_row):
                    time_val = exec_time_row.iloc[manip_idx]
                    if pd.notna(time_val) and str(time_val) not in ['None', 'end', 'time', '']:
                        time_str = str(time_val)
                        # Parse time format like "01:05" to seconds
                        if ':' in time_str:
                            try:
                                time_parts = time_str.split(':')
                                time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                                # Convert to relative normalized score (fastest = 1.0, slowest = 0.0)
                                time_score = (max_time - time_seconds) / time_range if time_range > 0 else 0.0
                                
                                # Get manipulation color and hand
                                manip_color = colors[manip_idx] if manip_idx < len(colors) else 'unknown'
                                manip_hand = hands[manip_idx] if manip_idx < len(hands) else 'unknown'
                                
                                parsed_data.append({
                                    'Policy': policy_name,
                                    'Trial': manip_idx + 1,
                                    'Color': manip_color,
                                    'Hand': manip_hand,
                                    'Task': 'Execution Time',
                                    'Score': time_score
                                })
                            except ValueError:
                                pass
        
        # Process Hand Back to Start Position data
        hand_back_row_idx = start_row + len(subtasks) + 3  # +3 for color, hand, and execution time rows
        if hand_back_row_idx < end_row:
            hand_back_row = raw_df.iloc[hand_back_row_idx, 1:]
            
            for manip_idx in range(valid_manipulations):
                if manip_idx < len(hand_back_row):
                    back_val = hand_back_row.iloc[manip_idx]
                    if pd.notna(back_val) and str(back_val) not in ['None', 'end', 'time', '']:
                        back_str = str(back_val)
                        
                        # Handle different value types - only parse numeric values
                        score_val = None
                        try:
                            # Try to parse as numeric value (0, 1, etc.)
                            score_val = float(back_str)
                        except ValueError:
                            # Skip non-numeric values like "00:00" or other time formats
                            continue
                        
                        if score_val is not None:
                            # Get manipulation color and hand
                            manip_color = colors[manip_idx] if manip_idx < len(colors) else 'unknown'
                            manip_hand = hands[manip_idx] if manip_idx < len(hands) else 'unknown'
                            
                            parsed_data.append({
                                'Policy': policy_name,
                                'Trial': manip_idx + 1,
                                'Color': manip_color,
                                'Hand': manip_hand,
                                'Task': 'Hand Back to Start Position',
                                'Score': score_val
                            })
    
    # Create DataFrame from parsed data
    df = pd.DataFrame(parsed_data)
    
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
    if not df.empty and 'Policy' in df.columns:
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
    if not df.empty and 'Policy' in df.columns:
        # Define task weights (all set to 1.0 for unweighted average)
        task_weights = {
            "Hand Move to Cube": 1.0,
            "Hand Grasp Cube": 1.0,
            "Hand Move to Box": 1.0,
            "Cube in Box": 1.0,
            "Hand Back to Start Position": 1.0,
            "Execution Time": 1.0,
            # Add can manipulation tasks for lighting test compatibility
            "Hand Move to Can": 1.0,
            "Hand Grasp Can": 1.0,
            "Hand Move to corrct Box": 1.0,
            "Can in corredt Box": 1.0,
            "Start position am ende": 1.0
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
                    trial_hand = policy_trial_data['Hand'].iloc[0]
                    
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
                            'Hand': trial_hand,
                            'Task': 'Total Score',
                            'Score': weighted_average
                        }])], ignore_index=True)
    
    return df, time_info


def create_radar_chart(df: pd.DataFrame, time_info: dict, output_dir: Path, include_total_score: bool = True) -> None:
    """Create a radar chart comparing all policies across subtasks for cube manipulation."""
    
    # Calculate mean scores per policy and task
    policy_stats = df.groupby(['Policy', 'Task'])['Score'].mean().unstack(fill_value=0)
    
    # Ensure all subtasks are present including new metrics
    subtasks = [
        "Hand Move to Cube",
        "Hand Grasp Cube", 
        "Hand Move to Box",
        "Cube in Box",
        "Hand Back to Start Position",
        "Execution Time"
    ]
    
    if include_total_score:
        subtasks.append("Total Score")
    
    for task in subtasks:
        if task not in policy_stats.columns:
            policy_stats[task] = 0
    
    policy_stats = policy_stats[subtasks]  # Reorder columns
    
    # Professional color scheme
    #thesis_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']
    thesis_colors = ['#3498db','#e377c2','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#34495e','#e67e22']
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

        # Thinner lines & smaller markers
        ax.plot(angles, values, 'o-', linewidth=1.8, label=format_policy_name(policy), color=color,
                markersize=6, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.4, alpha=0.9)

        ax.fill(angles, values, alpha=0.07, color=color)

        # Enhanced dynamic label positioning with consistent radius
        for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            # Hard-coded fix: show "0.0" labels for R-S policy where value is 0, except for "Move to Cube"
            should_show_label = value > 0.05
            if policy == "R-S" and value <= 0.05:
                task_name = subtasks[j] if j < len(subtasks) else "Unknown"
                if task_name != "Hand Move to Cube":  # Skip "Move to Cube" task
                    should_show_label = True
            
            if should_show_label:
                angle_deg = (angle * 180 / pi) % 360
                # Normalized policy offset in [-1, 1]
                norm_idx = (idx - (n_policies - 1) / 2) / ((n_policies - 1) / 2) if n_policies > 1 else 0.0

                # Stronger angle jitter at top/bottom to spread horizontally more
                if angle_deg <= 45 or (135 < angle_deg <= 225) or angle_deg >= 315:
                    angle_jitter = 0.45
                else:
                    angle_jitter = 0.30
                angle_shifted = angle + norm_idx * angle_jitter

                # Use consistent radius for all labels
                label_r = 1.1  # Fixed radius for all labels

                # Determine text alignment based on angle
                if angle_deg <= 45 or angle_deg >= 315:
                    ha, va = 'center', 'bottom'
                elif 45 < angle_deg <= 135:
                    ha, va = 'left', 'center'
                elif 135 < angle_deg <= 225:
                    ha, va = 'center', 'top'
                else:
                    ha, va = 'right', 'center'

                # Determine label text based on task type
                task_name = subtasks[j] if j < len(subtasks) else "Unknown"
                if task_name == "Execution Time":
                    if policy in time_info.get('policy_times', {}):
                        # Show actual time in minutes for policies with time data
                        actual_minutes = time_info['policy_times'][policy]['minutes']
                        label_text = f"{actual_minutes:.1f}min"
                    else:
                        # Show "No Time" for policies without time data (like R-S)
                        label_text = "No Time"
                else:
                    # Show normalized score for other tasks (including hard-coded 0.0 for R-S)
                    label_text = f"{value:.2f}"

                ax.text(angle_shifted, label_r, label_text, ha=ha, va=va, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.26', facecolor='white', edgecolor=color, alpha=0.9, linewidth=1.2),
                        zorder=10, clip_on=False)

    # Enhanced axis customization
    ax.set_xticks(angles[:-1])
    task_labels = [
        "Move to\nCube",
        "Grasp\nCube",
        "Move to\nBox",
        "Place Cube in\nBox",
        "Hand Back to\nStart Position",
        f"Execution\nTime\n({time_info['min_time_minutes']:.1f}-\n{time_info['max_time_minutes']:.1f} min)"
    ]
    
    if include_total_score:
        task_labels.append("Total\nScore")
        
    ax.set_xticklabels(task_labels, fontsize=14, fontweight='bold', ha='center')
    ax.tick_params(axis='x', pad=45)  # push all task labels outward

    # Increase radial limit to make room for outside labels
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=14, alpha=0.95, fontweight='medium')

    # Add radial grid lines at specific values
    for tick in [0.2,0.4,0.6,0.8,1.0]:
        ax.plot([0, 2*pi], [tick, tick], color='gray', alpha=0.22, linewidth=0.8)
    
    # Legend moved slightly left
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.05, -0.15), borderaxespad=0.0, frameon=True, fancybox=True, shadow=True,
                       fontsize=13, title='ACT Policies', title_fontsize=13)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    legend.get_title().set_fontweight('bold')

    # Titles
    fig.suptitle('Policy Performance Comparison on Place Cube in Box Task', x=0.23, y=1.0, size=18, fontweight='bold', color='#2c3e50', ha='left')
    fig.text(0.33, 0.975, 'Success Rate by Subtask (0.0 = Failure, 1.0 = Success)', 
             ha='left', va='top', fontsize=14, style='italic', color='#6c757d')

    plt.tight_layout(rect=[0.00, 0.00, 1.00, 0.97])
    plt.savefig(output_dir / 'radar_chart_policy_comparison.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')


def create_grouped_bar_plot(df: pd.DataFrame, time_info: dict, output_dir: Path) -> None:
    """Create beautiful grouped bar plots with error bars for each subtask for cube manipulation."""
    
    # Calculate statistics
    stats = df.groupby(['Policy', 'Task'])['Score'].agg(['mean', 'std', 'count']).reset_index()
    
    subtasks = [
        "Hand Move to Cube",
        "Hand Grasp Cube",
        "Hand Move to Box",
        "Cube in Box",
        "Hand Back to Start Position",
        "Execution Time"
    ]
    policies = stats['Policy'].unique()
    
    # Professional color scheme for thesis
    thesis_colors = ['#3498db','#e377c2','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#34495e','#e67e22']
    
    # Optimized for A4 format - taller and narrower with better aspect ratio
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), dpi=150)
    
    # Global styling
    fig.patch.set_facecolor('white')
    
    # Create a subplot for each subtask
    for task_idx, task in enumerate(subtasks):
        ax = axes[task_idx // 2, task_idx % 2]
        row_idx = task_idx // 2
        col_idx = task_idx % 2
        
        task_data = stats[stats['Task'] == task]
        
        x = np.arange(len(policies))
        means, stds = [], []
        
        for policy in policies:
            policy_data = task_data[task_data['Policy'] == policy]
            if len(policy_data) > 0:
                means.append(policy_data['mean'].iloc[0])
                std_val = policy_data['std'].iloc[0] if pd.notna(policy_data['std'].iloc[0]) else 0
                stds.append(std_val)
            else:
                means.append(0); stds.append(0)
        
        # Create beautiful bars with enhanced styling
        bars = ax.bar(x, means, yerr=stds, capsize=8,
                     color=[thesis_colors[i % len(thesis_colors)] for i in range(len(policies))],
                     alpha=0.85, edgecolor='white', linewidth=2,
                     error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50', 'alpha': 0.8})
        
        # Add gradient effect to bars
        for i, bar in enumerate(bars):
            gradient = plt.Rectangle((bar.get_x(), 0), bar.get_width(), bar.get_height(),
                                   facecolor=thesis_colors[i % len(thesis_colors)], alpha=0.30, edgecolor='none')
            ax.add_patch(gradient)
        
        # Enhanced subplot styling
        ax.set_facecolor('#fafafa')
        ax.grid(axis='y', linestyle='--', alpha=0.45, linewidth=1, color='#bdc3c7')
        ax.set_axisbelow(True)
        
        # Customize subplot titles with better formatting
        task_title = task.replace(' to ', ' to\n') if len(task) > 20 else task
        ax.set_title(f'{task_title}', fontsize=24, fontweight='bold', pad=25, color='#2c3e50')
        
        # Only show y-axis label and ticks for left column (shared for the row)
        if col_idx == 0:  # Left column
            ax.set_ylabel('Success Rate', fontsize=22, fontweight='bold', color='#2c3e50')
            ax.tick_params(axis='y', labelsize=18, colors='#34495e', width=2, length=6)
        else:  # Right column - hide y-axis labels but keep ticks
            ax.tick_params(axis='y', labelsize=0, width=2, length=6)
        
        # Remove x-axis labels to save space - policy info is in the legend
        ax.set_xticks(x)
        ax.set_xticklabels([])  # No individual policy labels
        ax.tick_params(axis='x', length=0)  # Hide x-axis tick marks
        
        # Set consistent y-axis limits
        ax.set_ylim(0, 1.42)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
        
        # Add a horizontal line indicating 100% success rate
        ax.axhline(y=1.0, color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5)
        
        # Add value labels on bars with enhanced styling
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            label_y = height + std + 0.05
            if task == "Execution Time":
                if policies[i] in time_info.get('policy_times', {}):
                    label_text = f"{time_info['policy_times'][policies[i]]['minutes']:.1f}min"
                else:
                    # Handle special case for policies without time data (like R-S)
                    label_text = "No Time"
                rotation = 90  # Tilt execution time labels 90 degrees
            else:
                label_text = f"{mean:.2f}"  # Two digits after decimal
                rotation = 0  # Keep other labels horizontal
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y, label_text, ha='center', va='bottom', 
                    fontsize=18, fontweight='bold', color='#2c3e50', rotation=rotation,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=thesis_colors[i % len(thesis_colors)], 
                             alpha=0.92, linewidth=1.4)
                    )
        
        # Add horizontal reference lines for common thresholds
        for threshold, color, style in [(0.5, '#e74c3c', '--'), (0.8, '#27ae60', ':')]:
            ax.axhline(y=threshold, color=color, linestyle=style, alpha=0.55, linewidth=1.7)
        
        # Add visual separation between subplots in the same row
        if col_idx == 0:  # Left subplot - add right border
            ax.spines['right'].set_edgecolor('#2c3e50')
            ax.spines['right'].set_linewidth(3)
        else:  # Right subplot - add left border  
            ax.spines['left'].set_edgecolor('#2c3e50')
            ax.spines['left'].set_linewidth(3)
            
        # Style other borders
        for spine_name in ['top', 'bottom']:
            ax.spines[spine_name].set_edgecolor('#bdc3c7')
            ax.spines[spine_name].set_linewidth(1.6)
        if col_idx == 0:
            ax.spines['left'].set_edgecolor('#bdc3c7')
            ax.spines['left'].set_linewidth(1.6)
        else:
            ax.spines['right'].set_edgecolor('#bdc3c7')
            ax.spines['right'].set_linewidth(1.6)
    
    # Professional main title with subtitle
    fig.suptitle('ACT Policy Performance Analysis: Grasp Cube and Place in Box Task', 
                fontsize=28, fontweight='bold', y=0.985, color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.965, 'Mean Success Rate ± Standard Deviation by Subtask', 
             ha='center', va='top', fontsize=20, style='italic', color='#7f8c8d')
    
    # Create combined legend with policies and reference lines
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    
    # Policy legend elements
    policy_legend_elements = []
    for i, policy in enumerate(policies):
        policy_legend_elements.append(
            Rectangle((0, 0), 1, 1, facecolor=thesis_colors[i % len(thesis_colors)], 
                     alpha=0.85, edgecolor='white', linewidth=1,
                     label=format_policy_name(policy))
        )
    
    # Reference lines legend elements  
    reference_legend_elements = [
        Line2D([0], [0], color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5, label='100% Success'),
        Line2D([0], [0], color='#e74c3c', linestyle='--', alpha=0.6, linewidth=1.7, label='50% Success'),
        Line2D([0], [0], color='#27ae60', linestyle=':', alpha=0.6, linewidth=1.7, label='80% Success')
    ]
    
    # Combine all legend elements
    all_legend_elements = policy_legend_elements + reference_legend_elements
    
    # Position legend across the bottom in three rows - closer to plots with bigger text
    legend = fig.legend(handles=all_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04),
                       frameon=True, fancybox=True, shadow=True, fontsize=20, 
                       ncol=(len(all_legend_elements) + 2) // 3, columnspacing=1.5, handletextpad=0.8)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    
    # Professional layout with proper spacing optimized for A4 - reduced bottom margin for closer legend
    plt.tight_layout(rect=[0, 0.11, 1, 0.96])
    
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')


def create_hand_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create analysis comparing left vs right hand performance."""
    
    # Filter data that has hand information
    hand_data = df[df['Hand'].isin(['left', 'right'])]
    
    if hand_data.empty:
        print("No hand data found")
        return
    
    # Professional styling optimized for A4 format - narrower but taller
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=150)
    
    # 1. Overall performance by hand
    hand_stats = hand_data.groupby('Hand')['Score'].agg(['mean', 'std', 'count'])
    
    colors = ['#3498db', '#e74c3c']
    x = np.arange(len(hand_stats.index))
    
    bars = ax1.bar(x, hand_stats['mean'], yerr=hand_stats['std'], capsize=12, color=colors[:len(hand_stats)], 
                   alpha=0.85, edgecolor='white', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Left Hand', 'Right Hand'], fontsize=22)
    # Only show y-axis label for left subplot
    ax1.set_ylabel('Success Rate', fontsize=26, fontweight='bold', color='#2c3e50')
    ax1.set_title('Overall Performance by Hand', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
    ax1.set_ylim(0, 1.22)
    ax1.grid(axis='y', alpha=0.4)
    ax1.tick_params(axis='y', labelsize=20, colors='#34495e', width=2, length=6)
    ax1.tick_params(axis='x', labelsize=22)
    
    # Add value labels with bigger text
    for i, (bar, mean, std) in enumerate(zip(bars, hand_stats['mean'], hand_stats['std'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.035, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=20,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[i % len(colors)], 
                         alpha=0.92, linewidth=1.4))
    
    # 2. Performance by hand and task
    task_hand_stats = hand_data.groupby(['Task', 'Hand'])['Score'].mean().unstack(fill_value=0)
    
    # Filter to main tasks only
    main_tasks = ["Hand Move to Cube","Hand Grasp Cube","Hand Move to Box","Cube in Box","Hand Back to Start Position","Execution Time"]
    task_hand_stats = task_hand_stats.loc[task_hand_stats.index.isin(main_tasks)]
    
    x2 = np.arange(len(task_hand_stats.index))
    width = 0.38
    
    if 'left' in task_hand_stats.columns and 'right' in task_hand_stats.columns:
        bars1 = ax2.bar(x2 - width/2, task_hand_stats['left'], width, color='#3498db', 
                       alpha=0.82, edgecolor='white', linewidth=1)
        bars2 = ax2.bar(x2 + width/2, task_hand_stats['right'], width, color='#e74c3c', 
                       alpha=0.82, edgecolor='white', linewidth=1)
        
        ax2.set_xticks(x2)
        ax2.set_xticklabels([task.replace(' ', '\n') for task in task_hand_stats.index], fontsize=20)
        # Hide y-axis label for right subplot but keep ticks
        ax2.tick_params(axis='y', labelsize=0, width=2, length=6)
        ax2.set_title('Performance by Task and Hand', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
        ax2.set_ylim(0, 1.27)
        ax2.grid(axis='y', alpha=0.4)
        ax2.tick_params(axis='x', labelsize=20)
        
        # Add visual separation - right border for top row combination
        ax2.spines['left'].set_edgecolor('#2c3e50')
        ax2.spines['left'].set_linewidth(3)
    
    # 3. Performance by hand and color
    if len(hand_data['Color'].unique()) > 1:
        color_hand_stats = hand_data.groupby(['Color', 'Hand'])['Score'].mean().unstack(fill_value=0)
        
        x3 = np.arange(len(color_hand_stats.index))
        
        if 'left' in color_hand_stats.columns and 'right' in color_hand_stats.columns:
            bars3 = ax3.bar(x3 - width/2, color_hand_stats['left'], width, color='#3498db', 
                           alpha=0.82, edgecolor='white', linewidth=1)
            bars4 = ax3.bar(x3 + width/2, color_hand_stats['right'], width, color='#e74c3c', 
                           alpha=0.82, edgecolor='white', linewidth=1)
            
            for bar, value in zip(bars3, color_hand_stats['left']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{value:.2f}', 
                        ha='center', va='bottom', fontsize=18, fontweight='bold')
            
            for bar, value in zip(bars4, color_hand_stats['right']):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{value:.2f}', 
                        ha='center', va='bottom', fontsize=18, fontweight='bold')
            
            ax3.set_xticks(x3)
            ax3.set_xticklabels([color.title() for color in color_hand_stats.index], fontsize=22)
            ax3.set_ylabel('Success Rate', fontsize=26, fontweight='bold', color='#2c3e50')
            ax3.set_title('Performance by Hand and Color', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
            ax3.set_ylim(0, 1.22)
            ax3.grid(axis='y', alpha=0.4)
            ax3.tick_params(axis='y', labelsize=20, colors='#34495e', width=2, length=6)
            ax3.tick_params(axis='x', labelsize=22)
    
    # 4. Hand usage distribution
    hand_counts = hand_data['Hand'].value_counts()
    colors_pie = ['#3498db', '#e74c3c']
    
    wedges, texts, autotexts = ax4.pie(hand_counts.values, labels=None,  # Remove individual labels 
                                      autopct='%1.1f%%', colors=colors_pie, startangle=90, textprops={'fontsize': 20})
    ax4.set_title('Hand Usage Distribution', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
    
    # Enhance pie chart text with bigger fonts
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(20)
    
    # Add beautiful separations between plots
    ax1.spines['right'].set_edgecolor('#2c3e50')
    ax1.spines['right'].set_linewidth(3)
    ax3.spines['right'].set_edgecolor('#2c3e50')
    ax3.spines['right'].set_linewidth(3)
    ax4.spines['left'].set_edgecolor('#2c3e50')
    ax4.spines['left'].set_linewidth(3)
    
    # Style other borders for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        for spine_name in ['top', 'bottom']:
            ax.spines[spine_name].set_edgecolor('#bdc3c7')
            ax.spines[spine_name].set_linewidth(1.6)
        # Left and right borders for outer edges
        if ax in [ax1, ax3]:  # Left column
            ax.spines['left'].set_edgecolor('#bdc3c7')
            ax.spines['left'].set_linewidth(1.6)
        if ax in [ax2, ax4]:  # Right column
            ax.spines['right'].set_edgecolor('#bdc3c7')
            ax.spines['right'].set_linewidth(1.6)
    
    # Create hand legend at bottom right under hand distribution
    from matplotlib.patches import Rectangle
    legend_elements = []
    for i, hand in enumerate(['left', 'right']):
        if hand in hand_data['Hand'].values:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=colors_pie[i], alpha=0.82, 
                         edgecolor='white', linewidth=1, label=f'{hand.title()} Hand')
            )
    
    # Position legend under the pie chart with bigger text
    if legend_elements:
        legend = fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.02, 0.02),
                           frameon=True, fancybox=True, shadow=True, fontsize=22,
                           title='Hand Usage', title_fontsize=24, ncol=1)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        legend.get_frame().set_linewidth(1.4)
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color('#2c3e50')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_dir / 'hand_analysis.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_color_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create focused analysis for different cube colors (red, green, black)."""
    
    # Filter data that has color information
    color_data = df[df['Color'].isin(['red', 'green', 'black'])]
    
    if color_data.empty:
        print("No color data found")
        return
    
    # Professional styling optimized for A4 format - narrower but taller
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), dpi=150)
    
    # 1. Overall performance by color
    color_stats = color_data.groupby('Color')['Score'].agg(['mean', 'std', 'count'])
    
    # Use color-appropriate colors for visualization
    color_map = {'red': '#e74c3c', 'green': '#2ecc71', 'black': '#34495e'}
    bar_colors = [color_map.get(color, '#95a5a6') for color in color_stats.index]
    
    x = np.arange(len(color_stats.index))
    
    bars = ax1.bar(x, color_stats['mean'], yerr=color_stats['std'], capsize=12, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([color.title() + ' Cubes' for color in color_stats.index], fontsize=22)
    # Only show y-axis label for left subplot
    ax1.set_ylabel('Success Rate', fontsize=26, fontweight='bold', color='#2c3e50')
    ax1.set_title('Performance by Cube Color', fontsize=22, fontweight='bold', pad=25, color='#2c3e50')
    ax1.set_ylim(0, 1.22)
    ax1.grid(axis='y', alpha=0.4)
    ax1.tick_params(axis='y', labelsize=20, colors='#34495e', width=2, length=6)
    ax1.tick_params(axis='x', labelsize=22)
    
    # Add value labels with bigger text
    for i, (bar, mean, std) in enumerate(zip(bars, color_stats['mean'], color_stats['std'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.035, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=20,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=bar_colors[i % len(bar_colors)], 
                         alpha=0.92, linewidth=1.4))
    
    # 2. Performance by color and task
    task_color_stats = color_data.groupby(['Task', 'Color'])['Score'].mean().unstack(fill_value=0)
    
    # Filter to main tasks only
    main_tasks = ["Hand Move to Cube","Hand Grasp Cube","Hand Move to Box","Cube in Box","Hand Back to Start Position"]
    task_color_stats = task_color_stats.loc[task_color_stats.index.isin(main_tasks)]
    
    x2 = np.arange(len(task_color_stats.index))
    width = 0.25
    
    available_colors = [col for col in ['red', 'green', 'black'] if col in task_color_stats.columns]
    
    for i, color in enumerate(available_colors):
        offset = (i - len(available_colors)/2 + 0.5) * width
        ax2.bar(x2 + offset, task_color_stats[color], width, color=color_map[color], alpha=0.80, edgecolor='white', linewidth=1)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels([task.replace(' ', '\n') for task in task_color_stats.index], fontsize=20)
    # Hide y-axis label for right subplot but keep ticks
    ax2.tick_params(axis='y', labelsize=0, width=2, length=6)
    ax2.set_title('Performance by Task and Cube Color', fontsize=22, fontweight='bold', pad=25, color='#2c3e50')
    ax2.set_ylim(0, 1.27)
    ax2.grid(axis='y', alpha=0.4)
    ax2.tick_params(axis='x', labelsize=20)
    
    # Add visual separation - right border for top row combination
    ax2.spines['left'].set_edgecolor('#2c3e50')
    ax2.spines['left'].set_linewidth(3)
    
    # 3. Performance by color and policy
    policy_color_stats = color_data.groupby(['Policy', 'Color'])['Score'].mean().unstack(fill_value=0)
    
    x3 = np.arange(len(policy_color_stats.index))
    
    for i, color in enumerate(available_colors):
        offset = (i - len(available_colors)/2 + 0.5) * width
        if color in policy_color_stats.columns:
            ax3.bar(x3 + offset, policy_color_stats[color], width, color=color_map[color], alpha=0.80, edgecolor='white', linewidth=1)
    
    ax3.set_xticks(x3)
    ax3.set_xticklabels([format_policy_name(policy) for policy in policy_color_stats.index], rotation=45, ha='right', fontsize=20)
    # Color the tick labels to match policy colors
    thesis_colors = ['#3498db','#e377c2','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#34495e','#e67e22']
    for i, tick in enumerate(ax3.get_xticklabels()):
        tick.set_color(thesis_colors[i % len(thesis_colors)])
        tick.set_fontweight('bold')
    
    ax3.set_ylabel('Success Rate', fontsize=26, fontweight='bold', color='#2c3e50')
    ax3.set_title('Performance by Policy and Cube Color', fontsize=28, fontweight='bold', pad=22, color='#2c3e50')
    ax3.set_ylim(0, 1.27)
    ax3.grid(axis='y', alpha=0.4)
    ax3.tick_params(axis='y', labelsize=20, colors='#34495e', width=2, length=6)
    ax3.tick_params(axis='x', labelsize=20)
    
    # 4. Color distribution
    color_counts = color_data['Color'].value_counts()
    pie_colors = [color_map.get(color, '#95a5a6') for color in color_counts.index]
    
    wedges, texts, autotexts = ax4.pie(color_counts.values, 
                                      labels=None,  # Remove individual labels from pie
                                      autopct='%1.1f%%', colors=pie_colors, startangle=90, textprops={'fontsize': 20})
    ax4.set_title('Cube Color Distribution', fontsize=22, fontweight='bold', pad=25, color='#2c3e50')
    
    # Enhance pie chart text with bigger fonts
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(20)
    
    # Add beautiful separations between plots
    ax1.spines['right'].set_edgecolor('#2c3e50')
    ax1.spines['right'].set_linewidth(3)
    ax3.spines['right'].set_edgecolor('#2c3e50')
    ax3.spines['right'].set_linewidth(3)
    ax4.spines['left'].set_edgecolor('#2c3e50')
    ax4.spines['left'].set_linewidth(3)
    
    # Style other borders for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        for spine_name in ['top', 'bottom']:
            ax.spines[spine_name].set_edgecolor('#bdc3c7')
            ax.spines[spine_name].set_linewidth(1.6)
        # Left and right borders for outer edges
        if ax in [ax1, ax3]:  # Left column
            ax.spines['left'].set_edgecolor('#bdc3c7')
            ax.spines['left'].set_linewidth(1.6)
        if ax in [ax2, ax4]:  # Right column
            ax.spines['right'].set_edgecolor('#bdc3c7')
            ax.spines['right'].set_linewidth(1.6)
    
    # Create cube color legend at bottom right under color distribution
    from matplotlib.patches import Rectangle
    legend_elements = []
    for color in ['red', 'green', 'black']:
        if color in available_colors:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color_map[color], alpha=0.80, 
                         edgecolor='white', linewidth=1, label=f'{color.title()} Cubes')
            )
    
    # Position legend under the pie chart with bigger text
    if legend_elements:
        legend = fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.05),
                           frameon=True, fancybox=True, shadow=True, fontsize=22,
                           title='Cube Colors', title_fontsize=24, ncol=1)
        legend.get_frame().set_facecolor('#f8f9fa')
        legend.get_frame().set_edgecolor('#dee2e6')
        legend.get_frame().set_linewidth(1.4)
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_color('#2c3e50')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_dir / 'color_analysis.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_total_score_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create focused analysis for Total Score performance."""
    
    # Filter for Total Score data
    total_score_data = df[df['Task'] == 'Total Score']
    
    if total_score_data.empty:
        print("No Total Score data found")
        return
    
    # Professional styling optimized for A4 format
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), dpi=150)
    
    # Global styling
    fig.patch.set_facecolor('white')
    
    # 1. Bar plot of total scores
    policy_stats = total_score_data.groupby('Policy')['Score'].agg(['mean', 'std', 'count'])
    
    colors = ['#3498db','#e377c2','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#34495e','#e67e22']
    x = np.arange(len(policy_stats.index))
    
    bars = ax1.bar(x, policy_stats['mean'], yerr=policy_stats['std'], capsize=12, color=colors[:len(policy_stats)], 
                   alpha=0.85, edgecolor='white', linewidth=2,
                   error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50', 'alpha': 0.8})
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([])  # Remove policy names from x-axis
    ax1.set_ylabel('Total Score', fontsize=26, fontweight='bold', color='#2c3e50')
    ax1.set_title('Total Policy Performance Score', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
    ax1.set_ylim(0, 1.30)
    ax1.grid(axis='y', linestyle='--', alpha=0.45, linewidth=1, color='#bdc3c7')
    ax1.set_facecolor('#fafafa')
    ax1.tick_params(axis='y', labelsize=20, colors='#34495e', width=2, length=6)
    ax1.tick_params(axis='x', labelsize=20)
    
    # Add horizontal reference lines for common thresholds
    for threshold, color, style in [(0.5, '#e74c3c', '--'), (0.8, '#27ae60', ':'), (1.0, '#27ae60', '-')]:
        ax1.axhline(y=threshold, color=color, linestyle=style, alpha=0.7, linewidth=2.5)
    
    # Add separate mean and std deviation labels
    for i, (bar, mean, std) in enumerate(zip(bars, policy_stats['mean'], policy_stats['std'])):
        bar_height = bar.get_height()
        policy_name = policy_stats.index[i]
        
        # Mean label on top of actual bar (horizontal) - moved up to see bar end
        ax1.text(bar.get_x() + bar.get_width()/2., bar_height + 0.025, f'{mean:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=18, rotation=0,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=colors[i % len(colors)], 
                         alpha=0.92, linewidth=1.4))
        
        # Standard deviation label on top of error bar - skip for R-S policy and make more beautiful
        if policy_name != 'R-S':  # Skip std label for R-S policy to avoid overlap
            error_bar_top = bar_height + std
            ax1.text(bar.get_x() + bar.get_width()/2., error_bar_top + 0.035, f'±{std:.2f}', 
                    ha='center', va='bottom', fontweight='medium', fontsize=16, rotation=0,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#95a5a6', 
                             alpha=0.90, linewidth=1.2),
                    color="#000000")
    
    # 2. Color-based performance for total score
    if len(total_score_data['Color'].unique()) > 1:
        color_policy_stats = total_score_data.groupby(['Policy', 'Color'])['Score'].mean().unstack(fill_value=0)
        
        # Get available colors
        available_colors = [col for col in ['red', 'green', 'black'] if col in color_policy_stats.columns]
        
        if available_colors:
            # Add averaged totals row
            policies_list = list(color_policy_stats.index)
            
            # Calculate average across all policies for each color
            color_averages = {color: color_policy_stats[color].mean() for color in available_colors}
            
            # Create extended data including the average
            extended_policies = policies_list + ['Average']
            color_values = {color: list(color_policy_stats[color]) + [color_averages[color]] for color in available_colors}
            
            x2 = np.arange(len(extended_policies))
            width = 0.8 / len(available_colors)
            
            # Color mapping
            color_map = {'red': '#e74c3c', 'green': '#2ecc71', 'black': '#34495e'}
            
            # Plot bars for each color - no value labels
            for i, color in enumerate(available_colors):
                offset = (i - len(available_colors)/2 + 0.5) * width
                bars = ax2.bar(x2 + offset, color_values[color], width, 
                              label=f'{color.title()} Cubes', color=color_map.get(color, '#95a5a6'), 
                              alpha=0.85, edgecolor='white', linewidth=2)
            
            # Move x-labels to the right by adjusting tick positions
            ax2.set_xticks(x2 + 0.5)  # Shift ticks slightly to the right
            # Use colored ticks for policies, black for average
            tick_colors = []
            for i, policy in enumerate(extended_policies):
                if policy == 'Average':
                    tick_colors.append('#2c3e50')  # Dark color for average
                else:
                    tick_colors.append(colors[i % len(colors)])
            
            ax2.set_xticklabels(extended_policies, rotation=38, ha='right', fontsize=20)
            # Color the tick labels
            for i, (tick, color) in enumerate(zip(ax2.get_xticklabels(), tick_colors)):
                tick.set_color(color)
                tick.set_fontweight('bold')
            
            # Add a visual separator before the average column
            ax2.axvline(x=len(policies_list) - 0.5, color='#2c3e50', linestyle='--', alpha=0.7, linewidth=2)
        
        # Hide y-axis label for right subplot but keep ticks
        ax2.tick_params(axis='y', labelsize=0, width=2, length=6)
        ax2.set_title('Total Score by Cube Color', fontsize=25, fontweight='bold', pad=25, color='#2c3e50')
        
        # Make legend smaller and move it a bit more to the left
        legend_obj = ax2.legend(fontsize=18, loc='upper right', bbox_to_anchor=(0.50, 0.99), frameon=True, fancybox=True, shadow=True, 
                               title='Cube Colors', title_fontsize=20)
        legend_obj.get_frame().set_facecolor('#f8f9fa')
        legend_obj.get_frame().set_edgecolor('#dee2e6')
        legend_obj.get_frame().set_linewidth(1.4)
        legend_obj.get_title().set_fontweight('bold')
        legend_obj.get_title().set_color('#2c3e50')
        
        ax2.set_ylim(0, 1.30)
        ax2.grid(axis='y', linestyle='--', alpha=0.45, linewidth=1, color='#bdc3c7')
        ax2.set_facecolor('#fafafa')
        ax2.tick_params(axis='x', labelsize=20)
        
        # Add horizontal reference lines for common thresholds
        for threshold, color, style in [(0.5, '#e74c3c', '--'), (0.8, '#27ae60', ':'), (1.0, '#27ae60', '-')]:
            ax2.axhline(y=threshold, color=color, linestyle=style, alpha=0.7, linewidth=2.5)
    
    # Add beautiful separations between plots
    ax1.spines['right'].set_edgecolor('#2c3e50')
    ax1.spines['right'].set_linewidth(3)
    ax2.spines['left'].set_edgecolor('#2c3e50')
    ax2.spines['left'].set_linewidth(3)
    
    # Style other borders
    for ax in [ax1, ax2]:
        for spine_name in ['top', 'bottom']:
            ax.spines[spine_name].set_edgecolor('#bdc3c7')
            ax.spines[spine_name].set_linewidth(1.6)
    # Left and right borders for outer edges
    ax1.spines['left'].set_edgecolor('#bdc3c7')
    ax1.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_edgecolor('#bdc3c7')
    ax2.spines['right'].set_linewidth(1.6)
    
    # Create policy legend at bottom left
    from matplotlib.patches import Rectangle
    policy_legend_elements = []
    for i, policy in enumerate(policy_stats.index):
        policy_legend_elements.append(
            Rectangle((0, 0), 1, 1, facecolor=colors[i % len(colors)], 
                     alpha=0.85, edgecolor='white', linewidth=1,
                     label=format_policy_name(policy))
        )
    
    # Add reference lines to legend
    from matplotlib.lines import Line2D
    reference_legend_elements = [
        Line2D([0], [0], color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5, label='100% Success'),
        Line2D([0], [0], color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2.5, label='50% Success'),
        Line2D([0], [0], color='#27ae60', linestyle=':', alpha=0.7, linewidth=2.5, label='80% Success')
    ]
    
    # Combine all legend elements
    all_legend_elements = policy_legend_elements + reference_legend_elements
    
    # Position legend at bottom left in four rows - moved further left and higher up
    legend = fig.legend(handles=all_legend_elements, loc='lower left', bbox_to_anchor=(-0.01, 0.16),
                       frameon=True, fancybox=True, shadow=True, fontsize=18, 
                       ncol=(len(all_legend_elements) + 3) // 4, columnspacing=1.5, handletextpad=0.8,
                       title='ACT Policies & Reference Lines', title_fontsize=20)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('#2c3e50')
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.98])  # More space for legend
    plt.savefig(output_dir / 'total_score_analysis.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def main():
    """Main function to run the cube policy analysis."""
    parser = argparse.ArgumentParser(description='Analyze cube manipulation policy performance')
    parser.add_argument('--csv_path', default="plot_wandb/cubes_policies.csv",help='Path to the cubes_policies.csv file')
    parser.add_argument('--output_dir', default='plot_wandb/plots/cube_analysis', help='Output directory for plots')
    parser.add_argument('--plots', nargs='+', choices=['radar', 'grouped_bar', 'hand_analysis', 'color_analysis', 'total_score'], default=['radar', 'grouped_bar', 'hand_analysis', 'color_analysis', 'total_score'], help='Which plots to generate')
    parser.add_argument('--no_total_score', action='store_true', help='Exclude total score from radar chart')
    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(exist_ok=True)
    print("Parsing CSV data...")
    df, time_info = parse_csv_data(args.csv_path)
    print(f"Loaded {len(df)} data points")
    print(f"Policies: {df['Policy'].unique()}")
    print(f"Tasks: {df['Task'].unique()}")
    print(f"Colors: {df['Color'].unique()}")
    print(f"Hands: {df['Hand'].unique()}")
    if 'radar' in args.plots:
        print("Creating radar chart..."); create_radar_chart(df, time_info, output_dir, include_total_score=not args.no_total_score)
    if 'grouped_bar' in args.plots:
        print("Creating grouped bar plots..."); create_grouped_bar_plot(df, time_info, output_dir)
    if 'hand_analysis' in args.plots:
        print("Creating hand analysis..."); create_hand_analysis(df, output_dir)
    if 'color_analysis' in args.plots:
        print("Creating color analysis..."); create_color_analysis(df, output_dir)
    if 'total_score' in args.plots:
        print("Creating total score analysis..."); create_total_score_analysis(df, output_dir)
    print(f"Analysis complete! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
