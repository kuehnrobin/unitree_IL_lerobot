#!/usr/bin/env python3
"""
Pareto-style plots for ACT policy performance on can sorting and cube in box tasks.
Plots execution time vs. overall success rate (sum of subtasks, excluding time) for each trial.
Each policy is shown with a unique color, individual trials as points, and mean as a star.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from pathlib import Path
from typing import Tuple, List, Dict
import argparse


# Professional color palette for policies (matching the existing analysis scripts)
POLICY_COLORS = [
    '#3498db',  # Blue
    '#4b0082',  # Indigo
    '#228b22',  # Forest Green
    '#dc143c',  # Crimson
    '#ffd700',  # Gold
    '#ff6600',  # Bright Orange
    '#00bcd4',  # Cyan
    '#e377c2',  # Pink
    '#8b4513',  # Saddle Brown
    '#34495e',  # Slate Gray
]

# Policy name mappings for combining similar policies
# For can sorting: combine R-A-AUG into R-A
CAN_POLICY_COMBINE = {
    'R-A-AUG': 'R-A',
}

# For cubes: combine R-S_LWA-PV_AT_A-NH into R-S_LWA-PV_AT_A
CUBE_POLICY_COMBINE = {
    'R-S_LWA-PV_AT_A-NH': 'R-S_LWA-PV_AT_A',
}


def combine_policies(df: pd.DataFrame, policy_map: Dict[str, str]) -> pd.DataFrame:
    """Combine policies based on the provided mapping."""
    df = df.copy()
    df['Policy'] = df['Policy'].apply(lambda x: policy_map.get(x, x))
    return df


def format_policy_name(policy_name: str) -> str:
    """
    Format policy name to display subscripts correctly using matplotlib formatting.
    Converts underscore notation so only the next letter after underscore becomes subscript.
    Examples: 'S_LWA' -> 'S$_L$WA', 'R-S_LWA' -> 'R-S$_L$WA'
    """
    result = ""
    i = 0
    
    while i < len(policy_name):
        if policy_name[i] == '_' and i + 1 < len(policy_name):
            subscript_char = policy_name[i + 1]
            result += f"$_{{{subscript_char}}}$"
            i += 2
        else:
            result += policy_name[i]
            i += 1
    
    return result


def parse_time_string(time_str) -> float:
    """Parse time string in format 'MM:SS' to seconds."""
    if pd.isna(time_str):
        return None
    time_str = str(time_str).strip()
    if time_str.lower() in ['none', 'end', 'time', '']:
        return None
    if ':' in time_str:
        try:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None


def parse_score(score_val) -> float:
    """Parse score value, handling comma decimals and invalid values."""
    if pd.isna(score_val):
        return None
    s = str(score_val).strip()
    if s.lower() in ['none', 'end', 'time', '']:
        return None
    try:
        s = s.replace(',', '.')
        val = float(s)
        # Clamp to valid range
        if val > 1.0 or val < 0.0:
            return None
        return val
    except ValueError:
        return None


def parse_can_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse can sorting CSV to extract individual trial data with execution time and success rates.
    
    The can sorting CSV has a specific structure:
    - 'Config ID' row marks the start of a policy section  
    - Next row contains policy name and color sequence for each manipulation
    - Following 4 rows contain task scores for: Hand Move to Can, Hand Grasp Can, 
      Hand Move to Correct Box, Can in Correct Box
    - Last row contains Return to Home Position success (0/1) and execution time (MM:SS)
    - Each trial block is 8 columns: 6 manipulations + 'end' + 'time'
    
    Returns DataFrame with columns: Policy, Trial, ExecutionTime (seconds), SuccessRate (0-1)
    """
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)
    
    # Canonical manipulation tasks
    canonical_tasks = [
        'Hand Move to Can',
        'Hand Grasp Can',
        'Hand Move to Correct Box',
        'Can in Correct Box'
    ]
    
    def normalize_task(label: str) -> str:
        """Normalize task label to canonical name."""
        l = ' '.join(str(label).strip().split())
        low = l.lower()
        if low.startswith('hand move to') and 'box' in low:
            return 'Hand Move to Correct Box'
        if low.startswith('hand move to') and 'can' in low:
            return 'Hand Move to Can'
        if 'grasp' in low and 'can' in low:
            return 'Hand Grasp Can'
        if 'can in' in low and 'box' in low:
            return 'Can in Correct Box'
        return l
    
    # Find all policy sections (rows after each 'Config ID')
    policy_sections = []
    for i in range(len(raw_df)):
        first_cell = str(raw_df.iloc[i, 0]).strip()
        if first_cell.lower() == 'config id':
            policy_row = i + 1
            if policy_row < len(raw_df):
                policy_name = str(raw_df.iloc[policy_row, 0]).strip().upper()
                policy_name = policy_name.replace('CANS_', '')
                if policy_name and policy_name.lower() != 'config id':
                    policy_sections.append((policy_row, policy_name))
    
    trials_data = []
    
    for pr, policy_name in policy_sections:
        # Color row is at policy row (pr) - columns 1 onwards
        color_row = raw_df.iloc[pr, 1:]
        
        # Return to Home Position row is at pr + 5 (after 4 task rows)
        # This row contains both success values (0/1) and time values (MM:SS)
        time_row_idx = pr + 5
        if time_row_idx >= len(raw_df):
            continue
        
        time_row = raw_df.iloc[time_row_idx, 1:]
        
        # Process trials in blocks of 8 columns
        # Each trial block: columns 0-5 are 6 manipulations, column 6 is 'end', column 7 is 'time'
        trial_num = 0
        
        # Iterate through trial blocks (8 columns per block)
        block_idx = 0
        while block_idx * 8 < len(color_row):
            block_start = block_idx * 8
            
            # Collect scores for this trial's manipulations (up to 6 per trial)
            trial_scores = []
            
            for manip_offset in range(6):
                col_idx = block_start + manip_offset
                if col_idx >= len(color_row):
                    break
                
                color_val = color_row.iloc[col_idx] if col_idx < len(color_row) else None
                color = str(color_val).strip().lower() if pd.notna(color_val) else ''
                
                # Stop at 'end' marker or empty
                if color in ['end', 'time', 'none', '']:
                    break
                
                # Only process valid colors (red or green)
                if color not in ['red', 'green']:
                    continue
                
                # Get scores for all 4 subtasks for this manipulation
                for task_offset in range(4):
                    task_row_idx = pr + 1 + task_offset
                    if task_row_idx >= len(raw_df):
                        continue
                    
                    task_row = raw_df.iloc[task_row_idx, 1:]
                    if col_idx < len(task_row):
                        score = parse_score(task_row.iloc[col_idx])
                        if score is not None:
                            trial_scores.append(score)
            
            # Find the time column for this trial (at position 7 in the block)
            time_col = block_start + 7
            exec_time = None
            if time_col < len(time_row):
                exec_time = parse_time_string(time_row.iloc[time_col])
            
            # Include trials with scores, even if execution time is missing
            if trial_scores:
                trial_num += 1
                # Calculate overall success rate (mean of all manipulation scores)
                success_rate = np.mean(trial_scores)
                
                trials_data.append({
                    'Policy': policy_name,
                    'Trial': trial_num,
                    'ExecutionTime': exec_time if exec_time is not None else 0,
                    'SuccessRate': success_rate,
                    'NumScores': len(trial_scores),
                    'HasExecutionTime': exec_time is not None
                })
            
            block_idx += 1
    
    return pd.DataFrame(trials_data)


def parse_cube_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse cube in box CSV to extract individual trial data with execution time and success rates.
    
    The cube CSV has a different structure than cans:
    - Policy row contains policy name and color sequence
    - Next row contains 'Hand Used:' and left/right for each manipulation
    - Following rows contain task scores for each individual manipulation
    - Execution Time row contains times in MM:SS format
    
    Returns DataFrame with columns: Policy, Trial, ExecutionTime (seconds), SuccessRate (0-1)
    """
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)
    
    # Cube manipulation subtasks
    subtasks = [
        'Hand Move to Cube',
        'Hand Grasp Cube', 
        'Hand Move to Box',
        'Cube in Box'
    ]
    
    # Find policy start rows (rows containing 'cubes_' or policy patterns)
    policy_sections = []
    for idx in range(len(raw_df)):
        first_cell = str(raw_df.iloc[idx, 0]).strip()
        first_cell_lower = first_cell.lower()
        # Match policy rows (start with cubes_ or R- pattern but not "Hand" or "Config")
        if ('cubes_' in first_cell_lower or 
            (first_cell_lower.startswith('r-') and 'video' not in first_cell_lower)):
            if not any(skip in first_cell_lower for skip in ['hand', 'config', 'execution', 'time']):
                policy_name = first_cell.replace('cubes_', '').replace('CUBES_', '').upper()
                policy_sections.append((idx, policy_name))
    
    trials_data = []
    
    for sec_idx, (start_row, policy_name) in enumerate(policy_sections):
        # Find end of this section
        end_row = policy_sections[sec_idx + 1][0] if sec_idx + 1 < len(policy_sections) else len(raw_df)
        
        # Color row is at start_row (same row as policy name)
        color_row = raw_df.iloc[start_row, 1:]
        colors = []
        for c in color_row:
            c_str = str(c).lower().strip() if pd.notna(c) else ''
            if c_str in ['red', 'green', 'black']:
                colors.append(c_str)
            else:
                colors.append(None)
        
        # Hand row is at start_row + 1
        hand_row_idx = start_row + 1
        if hand_row_idx >= end_row:
            continue
        
        # Find the execution time row
        exec_time_row_idx = None
        for row_idx in range(start_row + 2, min(start_row + 10, end_row)):
            first_cell = str(raw_df.iloc[row_idx, 0]).strip().lower()
            if 'execution' in first_cell or (first_cell == 'time'):
                exec_time_row_idx = row_idx
                break
        
        if exec_time_row_idx is None:
            continue
        
        exec_time_row = raw_df.iloc[exec_time_row_idx, 1:]
        
        # Count valid manipulations
        valid_manip_count = sum(1 for c in colors if c is not None)
        
        # Process each manipulation (trial)
        for manip_idx in range(valid_manip_count):
            color = colors[manip_idx] if manip_idx < len(colors) else None
            if color is None:
                continue
            
            # Get execution time for this manipulation
            exec_time = None
            if manip_idx < len(exec_time_row):
                exec_time = parse_time_string(exec_time_row.iloc[manip_idx])
            
            # Collect scores for all subtasks
            scores = []
            for task_offset, task_name in enumerate(subtasks):
                task_row_idx = start_row + 2 + task_offset  # +2 for policy and hand rows
                if task_row_idx >= end_row:
                    continue
                
                task_row = raw_df.iloc[task_row_idx, 1:]
                if manip_idx < len(task_row):
                    score = parse_score(task_row.iloc[manip_idx])
                    if score is not None:
                        scores.append(score)
            
            # Include trials with scores, even if execution time is missing
            if scores:
                success_rate = np.mean(scores)
                trials_data.append({
                    'Policy': policy_name,
                    'Trial': manip_idx + 1,
                    'ExecutionTime': exec_time if exec_time is not None else 0,
                    'SuccessRate': success_rate,
                    'NumScores': len(scores),
                    'HasExecutionTime': exec_time is not None
                })
    
    return pd.DataFrame(trials_data)


def create_pareto_plot(df: pd.DataFrame, task_name: str, output_dir: Path,
                       scale_type: str = 'normal') -> None:
    """
    Create a Pareto-style plot showing execution time vs success rate.
    
    Args:
        df: DataFrame with columns Policy, Trial, ExecutionTime, SuccessRate
        task_name: Name of the task (e.g., 'Can Sorting' or 'Cube in Box')
        output_dir: Directory to save the plots
        scale_type: 'normal' for linear axes, 'inverse_log' for inverse log transformation
    """
    if df.empty:
        print(f"No data available for {task_name}")
        return
    
    # Set up the figure with larger fonts
    # Rectangular plot with legend below
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')  # White background instead of grey
    
    # Get unique policies
    policies = sorted(df['Policy'].unique())
    
    # Create color map for policies
    policy_color_map = {policy: POLICY_COLORS[i % len(POLICY_COLORS)] 
                        for i, policy in enumerate(policies)}
    
    # Import Line2D for legend
    from matplotlib.lines import Line2D
    
    # Plot individual trials and mean for each policy
    legend_elements = []
    
    # First pass: draw colored background regions for each policy cluster
    # Only use trials WITH execution time for the convex hull
    for policy in policies:
        policy_data = df[(df['Policy'] == policy) & (df['HasExecutionTime'] == True)]
        color = policy_color_map[policy]
        
        if len(policy_data) >= 3:
            # Get points for convex hull - use the EXACT data points
            points = np.column_stack([
                policy_data['SuccessRate'].values * 100,  # Convert to percentage
                policy_data['ExecutionTime'].values / 60  # Convert to minutes
            ])
            
            try:
                hull = ConvexHull(points)
                # Create a polygon from hull vertices - NO expansion, use exact points
                hull_points = points[hull.vertices]
                
                polygon = Polygon(hull_points, closed=True, 
                                  facecolor=color, edgecolor=color,
                                  alpha=0.15, linewidth=2, linestyle='--', zorder=1)
                ax.add_patch(polygon)
            except Exception:
                # If convex hull fails (e.g., collinear points), skip the region
                pass
        elif len(policy_data) == 2:
            # For 2 points, draw a line connecting them with some width
            x_vals = policy_data['SuccessRate'].values * 100  # Convert to percentage
            y_vals = policy_data['ExecutionTime'].values / 60
            
            # Draw a thick line between the two points
            ax.plot(x_vals, y_vals, color=color, alpha=0.3, linewidth=8, 
                    solid_capstyle='round', zorder=1)
    
    # Second pass: plot points and mean stars
    has_no_time_trials = False  # Track if we have any trials without execution time
    
    for policy in policies:
        policy_data = df[df['Policy'] == policy]
        color = policy_color_map[policy]
        
        # Separate trials with and without execution time
        with_time = policy_data[policy_data['HasExecutionTime'] == True]
        without_time = policy_data[policy_data['HasExecutionTime'] == False]
        
        # Plot trials WITH execution time as circles
        if len(with_time) > 0:
            ax.scatter(
                with_time['SuccessRate'] * 100,  # Convert to percentage
                with_time['ExecutionTime'] / 60,  # Convert to minutes
                c=color,
                s=180,  # Larger dots
                alpha=0.7,
                edgecolors='white',
                linewidths=1.5,
                label=None,
                zorder=3
            )
        
        # Plot trials WITHOUT execution time as X markers at y-axis minimum
        # Use task-specific y-values to ensure visibility (at the axis start)
        if len(without_time) > 0:
            has_no_time_trials = True
            # Set y-value based on task type to match axis start
            if 'can' in task_name.lower():
                no_time_y = 2.0  # ~2 minutes for can sorting
            elif 'cube' in task_name.lower():
                no_time_y = 0.27  # ~0.27 minutes for cube in box
            else:
                no_time_y = 0.5  # Default fallback
            
            ax.scatter(
                without_time['SuccessRate'] * 100,  # Convert to percentage
                [no_time_y] * len(without_time),  # Plot at axis minimum
                c=color,
                s=200,
                marker='X',
                alpha=0.8,
                edgecolors='#2c3e50',
                linewidths=1.5,
                label=None,
                zorder=3
            )
        
        # Calculate mean using ALL trials (including those without execution time for success rate)
        mean_success = policy_data['SuccessRate'].mean() * 100  # Convert to percentage
        # For mean time, only use trials that have execution time
        if len(with_time) > 0:
            mean_time = with_time['ExecutionTime'].mean() / 60  # Convert to minutes
        else:
            mean_time = 0  # If no trials have time, place star at 0
        
        ax.scatter(
            [mean_success],
            [mean_time],
            c=color,
            s=700,  # Larger stars
            marker='*',
            edgecolors='#2c3e50',
            linewidths=1.5,
            alpha=1.0,
            zorder=4
        )
        
        # Create legend element (circle for trials + policy name with count)
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=18, label=f'{format_policy_name(policy)} (n={len(policy_data)})',
                   markeredgecolor='white', markeredgewidth=2)
        )
    
    # Add reference lines for mean time and success (using all data for success rate)
    # For time mean, only use trials with execution time
    trials_with_time = df[df['HasExecutionTime'] == True]
    if len(trials_with_time) > 0:
        mean_time_overall = trials_with_time['ExecutionTime'].mean() / 60
    else:
        mean_time_overall = 0
    mean_success_overall = df['SuccessRate'].mean() * 100  # All trials contribute to success rate
    
    ax.axhline(y=mean_time_overall, color='#e74c3c', linestyle='--', 
               alpha=0.5, linewidth=2.5, label='Mean Execution Time')
    ax.axvline(x=mean_success_overall, color='#27ae60', linestyle='--',
               alpha=0.5, linewidth=2.5, label='Mean Success Rate')
    
    # Apply axis scaling based on scale_type
    if scale_type == 'inverse_log':
        # Use a power transformation: x^3 spreads out values much more as they approach 100%
        # Higher exponent = more expansion near 100%, more compression near 0%
        # Minimum spacing at 0%, maximum spacing at 100%
        ax.set_xscale('function', functions=(
            lambda x: np.power(np.clip(x, 0.01, 100.05), 3),    # forward: strong spread for high values
            lambda x: np.power(np.clip(x, 0.01, None), 1/3)     # inverse
        ))
        
        # Use log scale for y-axis (time increases upward, lower times compressed at bottom)
        ax.set_yscale('log')
        
        # Set custom tick locations for y-axis to show actual minute values
        y_min_data = df['ExecutionTime'].min() / 60
        y_max_data = df['ExecutionTime'].max() / 60
        
        # Generate nice tick values in minutes (explicitly no 6)
        tick_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20]
        tick_values = [t for t in tick_values if t >= y_min_data * 0.8 and t <= y_max_data * 1.2]
        if len(tick_values) < 3:
            tick_values = [0.5, 1, 2, 5, 10]
        
        ax.set_yticks(tick_values, minor=False)
        ax.set_yticklabels([f'{t:.1f}' if t < 1 else f'{int(t)}' for t in tick_values])
        # Disable minor ticks that might show scientific notation
        ax.yaxis.set_minor_locator(plt.NullLocator())
        
        x_label = 'Success Rate (%)'
        y_label = 'Execution Time (minutes)'
    else:
        # Normal linear axes
        x_label = 'Success Rate (%)'
        y_label = 'Execution Time (minutes)'
    
    # Styling with bigger fonts
    ax.set_xlabel(x_label, fontsize=24, 
                  fontweight='bold', color='#2c3e50', labelpad=12)
    ax.set_ylabel(y_label, fontsize=24, 
                  fontweight='bold', color='#2c3e50', labelpad=12)
    
    # Set axis limits based on task type (now in percentage)
    if 'can' in task_name.lower():
        x_min = 35  # Start at 35% for can sorting
        y_min_task = 1.95  # Start y-axis at 1.95 minutes for can sorting
    elif 'cube' in task_name.lower():
        x_min = 0  # Start at 0% to see all data
        y_min_task = 0.255  # Start y-axis at 0.255 minutes for cube in box
    else:
        x_min = 0
        y_min_task = 0.1
    ax.set_xlim(x_min, 100.005)  # Tiny extension past 100%
    
    # Y-axis limits in minutes - use task-specific minimum
    y_max = (df[df['HasExecutionTime'] == True]['ExecutionTime'].max() / 60) * 1.15
    if scale_type == 'inverse_log':
        ax.set_ylim(y_min_task, max(y_max, 1))
    else:
        ax.set_ylim(0, max(y_max, 1))
    
    ax.tick_params(axis='both', labelsize=20, colors='#34495e', width=2, length=6)
    
    # Disable default grid
    ax.grid(False)
    
    # Add custom vertical grid lines for x-axis
    # 10% steps (thick lines): 10, 20, 30, ..., 100
    for x in range(0, 101, 10):
        if x >= x_min:
            ax.axvline(x=x, color='#bdc3c7', linestyle='-', linewidth=1.5, alpha=0.6, zorder=0)
    
    # 5% steps (thinner lines, no label): 5, 15, 25, ..., 95
    for x in range(5, 100, 10):
        if x >= x_min:
            ax.axvline(x=x, color='#bdc3c7', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
    
    # 1% steps between 90-100 (thin lines)
    for x in range(91, 100):
        ax.axvline(x=x, color='#bdc3c7', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)
    
    # Add horizontal grid lines for y-axis
    ax.yaxis.grid(True, alpha=0.4, linewidth=0.8, color='#bdc3c7', linestyle='-')
    
    # Set x-axis ticks: 10% steps + 95%
    major_ticks = [x for x in range(0, 101, 10) if x >= x_min]
    if 95 not in major_ticks:
        major_ticks.append(95)
    major_ticks.sort()
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f'{x}' for x in major_ticks])
    
    # Enhanced borders
    for spine in ax.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1.6)
    
    # Title with bigger font
    ax.set_title(f'{task_name} Task',
                 fontsize=28, fontweight='bold', color='#2c3e50', pad=20)
    
    # Add legend elements for reference lines and mean marker
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#7f8c8d',
                                  markersize=28, label='Policy Mean (★)', markeredgecolor='#2c3e50',
                                  markeredgewidth=1))
    legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='#7f8c8d',
                                  markersize=18, label='No Execution Time', markeredgecolor='#7f8c8d',
                                  markeredgewidth=3, linestyle='None'))
    legend_elements.append(Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=4,
                                  label='Mean Time', alpha=0.7))
    legend_elements.append(Line2D([0], [0], color='#27ae60', linestyle='--', linewidth=4,
                                  label='Mean Success', alpha=0.7))
    
    # Legend below the plot with 3 columns
    legend = ax.legend(handles=legend_elements, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.12),  # Position below plot with more space
                       frameon=True, fancybox=True, shadow=True, fontsize=18,
                       ncol=3,  # 3 columns
                       title='Policies', title_fontsize=20)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    legend.get_title().set_fontweight('bold')
    
    # Adjust layout to make room for legend below
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)  # More room for larger legend
    
    # Save figures with scale type in filename
    safe_name = task_name.lower().replace(' ', '_')
    suffix = f'_{scale_type}' if scale_type != 'normal' else ''
    plt.savefig(output_dir / f'pareto_{safe_name}{suffix}.png', 
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / f'pareto_{safe_name}{suffix}.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Saved Pareto plot ({scale_type}) for {task_name} to {output_dir}")
    plt.close()


def print_summary_statistics(df: pd.DataFrame, task_name: str) -> None:
    """Print summary statistics for the parsed data."""
    print(f"\n{'='*60}")
    print(f"Summary Statistics: {task_name}")
    print(f"{'='*60}")
    
    if df.empty:
        print("No data available")
        return
    
    print(f"\nTotal trials: {len(df)}")
    print(f"Policies: {sorted(df['Policy'].unique())}")
    
    print(f"\nPer-policy statistics:")
    print(f"{'Policy':<25} {'Trials':>8} {'Mean Time':>12} {'Mean Success':>14}")
    print(f"{'-'*59}")
    
    for policy in sorted(df['Policy'].unique()):
        policy_data = df[df['Policy'] == policy]
        n_trials = len(policy_data)
        mean_time = policy_data['ExecutionTime'].mean() / 60
        mean_success = policy_data['SuccessRate'].mean()
        print(f"{policy:<25} {n_trials:>8} {mean_time:>10.2f}m {mean_success:>14.2%}")


def save_statistics_to_markdown(can_df: pd.DataFrame, cube_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save average execution time and average overall success rate for each policy to a markdown file.
    """
    md_path = output_dir / 'policy_statistics.md'
    
    with open(md_path, 'w') as f:
        f.write("# Policy Performance Statistics\n\n")
        f.write("This file contains the average execution time and average overall success rate for each policy.\n\n")
        
        # Can Sorting Statistics
        f.write("## Can Sorting Task\n\n")
        if not can_df.empty:
            f.write("| Policy | Trials | Mean Execution Time (min) | Mean Success Rate |\n")
            f.write("|--------|--------|---------------------------|-------------------|\n")
            
            for policy in sorted(can_df['Policy'].unique()):
                policy_data = can_df[can_df['Policy'] == policy]
                n_trials = len(policy_data)
                mean_time = policy_data['ExecutionTime'].mean() / 60
                mean_success = policy_data['SuccessRate'].mean()
                f.write(f"| {policy} | {n_trials} | {mean_time:.2f} | {mean_success:.2%} |\n")
            
            # Overall statistics for cans
            f.write(f"\n**Overall Can Sorting Statistics:**\n")
            f.write(f"- Total trials: {len(can_df)}\n")
            f.write(f"- Mean execution time: {can_df['ExecutionTime'].mean() / 60:.2f} minutes\n")
            f.write(f"- Mean success rate: {can_df['SuccessRate'].mean():.2%}\n")
        else:
            f.write("*No data available for Can Sorting task.*\n")
        
        f.write("\n")
        
        # Cube in Box Statistics
        f.write("## Cube in Box Task\n\n")
        if not cube_df.empty:
            f.write("| Policy | Trials | Mean Execution Time (min) | Mean Success Rate |\n")
            f.write("|--------|--------|---------------------------|-------------------|\n")
            
            for policy in sorted(cube_df['Policy'].unique()):
                policy_data = cube_df[cube_df['Policy'] == policy]
                n_trials = len(policy_data)
                mean_time = policy_data['ExecutionTime'].mean() / 60
                mean_success = policy_data['SuccessRate'].mean()
                f.write(f"| {policy} | {n_trials} | {mean_time:.2f} | {mean_success:.2%} |\n")
            
            # Overall statistics for cubes
            f.write(f"\n**Overall Cube in Box Statistics:**\n")
            f.write(f"- Total trials: {len(cube_df)}\n")
            f.write(f"- Mean execution time: {cube_df['ExecutionTime'].mean() / 60:.2f} minutes\n")
            f.write(f"- Mean success rate: {cube_df['SuccessRate'].mean():.2%}\n")
        else:
            f.write("*No data available for Cube in Box task.*\n")
        
        f.write("\n---\n")
        f.write("*Generated by pareto_plots.py*\n")
    
    print(f"Saved statistics to {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Pareto plots for ACT policy analysis')
    parser.add_argument('--can-csv', type=str, default='can_policies_time.csv',
                        help='Path to can policies CSV file')
    parser.add_argument('--cube-csv', type=str, default='cubes_policies.csv',
                        help='Path to cube policies CSV file')
    parser.add_argument('--output-dir', type=str, default='../oscar',
                        help='Output directory for plots')
    parser.add_argument('--no-combine', action='store_true',
                        help='Disable policy combination (keep all policies separate)')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    can_csv = script_dir / args.can_csv
    cube_csv = script_dir / args.cube_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Parsing CSV files...")
    
    # Parse can sorting data
    if can_csv.exists():
        can_df = parse_can_csv(str(can_csv))
        # Apply policy combination for cans
        if not args.no_combine:
            can_df = combine_policies(can_df, CAN_POLICY_COMBINE)
        print_summary_statistics(can_df, "Can Sorting")
    else:
        print(f"Warning: Can CSV not found at {can_csv}")
        can_df = pd.DataFrame()
    
    # Parse cube in box data
    if cube_csv.exists():
        cube_df = parse_cube_csv(str(cube_csv))
        # Apply policy combination for cubes
        if not args.no_combine:
            cube_df = combine_policies(cube_df, CUBE_POLICY_COMBINE)
        print_summary_statistics(cube_df, "Cube in Box")
    else:
        print(f"Warning: Cube CSV not found at {cube_csv}")
        cube_df = pd.DataFrame()
    
    print("\nGenerating Pareto plots...")
    
    # Create plots with both scale types for each task
    scale_types = ['normal', 'inverse_log']
    
    if not can_df.empty:
        for scale_type in scale_types:
            create_pareto_plot(can_df, "Can Sorting", output_dir, scale_type=scale_type)
    
    if not cube_df.empty:
        for scale_type in scale_types:
            create_pareto_plot(cube_df, "Cube in Box", output_dir, scale_type=scale_type)
    
    # Save statistics to markdown file
    save_statistics_to_markdown(can_df, cube_df, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
