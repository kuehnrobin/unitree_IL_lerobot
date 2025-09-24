#!/usr/bin/env python3
"""
Analysis and visualization of ACT policy performance on can manipulation tasks.
This script reads results from can_policies.csv and creates comparative visualizations.
"""
from math import pi
from pathlib import Path
from typing import Tuple
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, json, warnings


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
    """Parse lighting test / can policies CSV into unified long-form DataFrame.

    Enhancements:
    - Support comma decimal separator (e.g. 0,5) by replacing with dot before float conversion.
    - Compute TWO total scores per policy & color: 'Total Score' (with Return Home) and
      'Total Score (No RH)' (excluding Return to Home Position component) using weighting:
        With RH: (Sum 4 manipulation task means + ReturnHome + Time) / 6
        Without RH: (Sum 4 manipulation task means + Time) / 5
      (Time is already normalized; ReturnHome is success rate.)
    - Color-specific totals use color-specific manipulation means + shared all-color Time and ReturnHome.
    """
    raw_df = pd.read_csv(csv_path, delimiter=';', header=None)

    # Canonical task names
    canonical_tasks = [
        'Hand Move to Can',
        'Hand Grasp Can',
        'Hand Move to Correct Box',
        'Can in Correct Box'
    ]

    # Map raw row labels to canonical
    def normalize_task(label: str) -> str:
        l = ' '.join(label.strip().split())  # collapse multiple spaces
        low = l.lower()
        # Robust matching for the frequently misspelled / spaced variants
        if low.startswith('hand move to') and 'box' in low:
            return 'Hand Move to Correct Box'
        if low.startswith('hand move to') and 'can' in low:
            return 'Hand Move to Can'
        if 'grasp' in low and 'can' in low:
            return 'Hand Grasp Can'
        if 'can in correct box' in low:
            return 'Can in Correct Box'
        if 'return to home position' in low:
            return 'Return to Home Position'
        return l

    # Locate policy sections (rows after each 'Config ID')
    policy_rows = []  # list of tuples (policy_row_index, policy_name)
    for i in range(len(raw_df)):
        first_cell = str(raw_df.iloc[i, 0]).strip()
        if first_cell.lower() == 'config id':
            policy_row = i + 1
            if policy_row < len(raw_df):
                policy_name = str(raw_df.iloc[policy_row, 0]).strip().upper().replace('CANS_', '')
                if policy_name and policy_name.lower() != 'config id':
                    policy_rows.append((policy_row, policy_name))

    if not policy_rows:
        raise ValueError('No policy rows detected. Check CSV format.')

    # Gather all times for normalization
    all_times = []
    for pr, _ in policy_rows:
        end_idx = pr + len(canonical_tasks) + 1
        if end_idx < len(raw_df):
            row = raw_df.iloc[end_idx, 1:]
            # Iterate over trial blocks (8 columns per trial)
            for start in range(0, len(row), 8):
                t_idx = start + 7
                if t_idx < len(row):
                    val = row.iloc[t_idx]
                    if pd.notna(val):
                        s = str(val).strip()
                        if ':' in s and s.lower() not in ['none', 'end', 'time', '']:
                            try:
                                mm, ss = s.split(':')
                                all_times.append(int(mm)*60 + int(ss))
                            except ValueError:
                                pass
    if all_times:
        min_time, max_time = min(all_times), max(all_times)
        time_range = max_time - min_time if max_time > min_time else 1
    else:
        min_time, max_time, time_range = 0, 900, 900

    parsed_rows = []

    for pr, policy_name in policy_rows:
        color_row = raw_df.iloc[pr, 1:]
        # Build colors list; only keep 'red' or 'green'
        colors = [c.lower().strip() if pd.notna(c) and str(c).lower().strip() in ['red','green'] else None for c in color_row]

        # Process the four skill tasks
        for t_idx in range(4):
            task_row_idx = pr + 1 + t_idx
            if task_row_idx >= len(raw_df):
                continue
            raw_label = str(raw_df.iloc[task_row_idx, 0])
            task_name = normalize_task(raw_label)
            if task_name not in canonical_tasks:
                continue
            row_vals = raw_df.iloc[task_row_idx, 1:]
            red_scores, green_scores, all_scores = [], [], []
            for col_idx, val in enumerate(row_vals):
                if col_idx < len(colors) and colors[col_idx] in ['red','green']:
                    if pd.notna(val):
                        s = str(val).strip().replace(',', '.')
                        if s.lower() in ['none','end','time','']:
                            continue
                        try:
                            num = float(s)
                        except ValueError:
                            continue
                        # clamp improbable >1 values if any (defensive)
                        if num > 1.0:
                            continue
                        all_scores.append(num)
                        if colors[col_idx] == 'red':
                            red_scores.append(num)
                        else:
                            green_scores.append(num)
            # Append aggregated means
            if all_scores:
                parsed_rows.append({'Policy':policy_name,'Trial':1,'Color':'all','Task':task_name,'Score':float(np.mean(all_scores))})
            if red_scores:
                parsed_rows.append({'Policy':policy_name,'Trial':1,'Color':'red','Task':task_name,'Score':float(np.mean(red_scores))})
            if green_scores:
                parsed_rows.append({'Policy':policy_name,'Trial':1,'Color':'green','Task':task_name,'Score':float(np.mean(green_scores))})

        # Return to Home Position row
        end_idx = pr + len(canonical_tasks) + 1
        if end_idx < len(raw_df):
            raw_label = str(raw_df.iloc[end_idx, 0])
            if normalize_task(raw_label) == 'Return to Home Position':
                vals = raw_df.iloc[end_idx, 1:]
                success, times = [], []
                for v in vals:
                    if pd.isna(v):
                        continue
                    s = str(v).strip()
                    if s.lower() in ['none','end','time','']:
                        continue
                    if ':' in s:
                        try:
                            mm, ss = s.split(':')
                            times.append(int(mm)*60 + int(ss))
                        except ValueError:
                            pass
                    else:
                        try:
                            sc = float(s.replace(',','.'))
                            if sc in [0.0,1.0]:
                                success.append(sc)
                        except ValueError:
                            pass
                if success:
                    parsed_rows.append({'Policy':policy_name,'Trial':1,'Color':'all','Task':'Return to Home Position','Score':float(np.mean(success))})
                if times:
                    avg_t = float(np.mean(times))
                    time_score = (max_time - avg_t)/time_range if time_range>0 else 0.0
                    parsed_rows.append({'Policy':policy_name,'Trial':1,'Color':'all','Task':'Execution Time','Score':time_score})

    df = pd.DataFrame(parsed_rows)

    # policy times map
    policy_times = {}
    for policy in df['Policy'].unique():
        r = df[(df['Policy']==policy)&(df['Task']=='Execution Time')&(df['Color']=='all')]
        if not r.empty:
            sc = r['Score'].iloc[0]
            actual = max_time - sc * time_range
            policy_times[policy] = {'seconds':actual,'minutes':actual/60.0}

    time_info = {'min_time_seconds':min_time,'max_time_seconds':max_time,'time_range_seconds':time_range,'min_time_minutes':min_time/60.0,'max_time_minutes':max_time/60.0,'policy_times':policy_times}

    # Compute total scores (with & without Return Home)
    tasks_four = canonical_tasks
    total_rows = []
    for policy in df['Policy'].unique():
        # required shared components
        home_row = df[(df['Policy']==policy)&(df['Task']=='Return to Home Position')&(df['Color']=='all')]
        time_row = df[(df['Policy']==policy)&(df['Task']=='Execution Time')&(df['Color']=='all')]
        home_score = home_row['Score'].iloc[0] if not home_row.empty else None
        time_score = time_row['Score'].iloc[0] if not time_row.empty else None
        for color_tag in ['all','red','green']:
            # collect manipulation task means for this color
            manip = []
            for t in tasks_four:
                r = df[(df['Policy']==policy)&(df['Task']==t)&(df['Color']==color_tag)]
                if r.empty:
                    manip = []
                    break
                manip.append(r['Score'].iloc[0])
            if len(manip)!=4 or time_score is None:
                continue
            # total without RH
            components_no = manip + [time_score]  # (sum manip + time)/5
            total_no = float(np.mean(components_no))
            std_no = float(np.std(components_no, ddof=1)) if len(components_no)>1 else 0.0
            total_rows.append({'Policy':policy,'Trial':1,'Color':color_tag,'Task':'Total Score (No RH)','Score':total_no,'Std':std_no})
            if home_score is not None:
                components_with = manip + [home_score, time_score]  # (sum manip + home + time)/6
                total_with = float(np.mean(components_with))
                std_with = float(np.std(components_with, ddof=1)) if len(components_with)>1 else 0.0
                total_rows.append({'Policy':policy,'Trial':1,'Color':color_tag,'Task':'Total Score','Score':total_with,'Std':std_with})
    if total_rows:
        totals_df = pd.DataFrame(total_rows)
        df = pd.concat([df, totals_df[['Policy','Trial','Color','Task','Score']]], ignore_index=True)

    return df, time_info

#colors = ['#3498db','#e377c2','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#34495e','#e67e22']
# Uniform policy color palette (last orange replaced with pink)
# POLICY_COLORS = [
#     '#3498db',  # Blue R-A    # Passt
#     '#00bcd4',  # Red R-A-AUG # Neue Farbe z.B. cyan?
#     '#e74c3c',  # Red R-A-AUG  # Raus weil R-S_LWA-P nicht verwendet
#     '#2ecc71',  # Green R-A-P # Raus weil R-S_LWA-PV_AT_A nicht verwendet
#     '#f39c12',  # Orange R-SW # Raus weil R-S_LWA-PV_AT_A nicht verwendet
#     '#9b59b6',  # Purple R-S_LWA
#     '#1abc9c',  # Teal R-WA # Muss purple werden
#     '#34495e',  # Slate Grey R-WA-P # Muss Teal werden
#     '#ff69b4',  # Pink R-WA-PV_AT_A # Pink Raus weil R-S nicht verwendet # Den Eintrag zu Slate gray
#     '#00bcd4',  # Cyan R-W_RA
# ]
POLICY_COLORS = [
    '#3498db',  # Blue R-A    # Passt
    '#4b0082',  # Indigo R-A-AUG # Bright yellow-gold, distinct from all other colors
    '#228b22',  # Forest Green R-A-P # Professional green tone
    '#dc143c',  # Crimson R-SW # Deep red, distinct from other reds
    '#ffd700',  # Gold  R-S_LWA # Deep purple-blue, distinct from purple
    '#9b59b6',  # Purple R-WA # Passt
    '#1abc9c',  # Teal R-WA-P # Passt
    '#34495e',  # Slate Gray R-WA-PV_AT_A # Passt
    '#ff6347',  # Tomato R-W_RA # Orange-red, easily distinguishable
]

POLICY_COLOR_MAP = {}


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
        # Use universal color map
        color = POLICY_COLOR_MAP.get(policy, POLICY_COLORS[idx % len(POLICY_COLORS)])
        values = scores.tolist()
        values += values[:1]
        
        # Enhanced styling with thinner lines and smaller markers for professional appearance
        ax.plot(angles, values, 'o-', linewidth=1.8, label=format_policy_name(policy), color=color,
               markersize=6, markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=1.4, alpha=0.9)
        ax.fill(angles, values, alpha=0.07, color=color)

        # Enhanced dynamic label positioning with consistent radius
        for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            # Enhanced condition for showing labels with better logic for zero values
            should_show_label = value > 0.05
            if policy.startswith('R-S') and value <= 0.05:
                task_name = subtasks[j] if j < len(subtasks) else "Unknown"
                if task_name != "Hand Move to Can":  # Skip first task for R-S policy
                    should_show_label = True
            
            if should_show_label:
                angle_deg = (angle * 180 / pi) % 360
                # Normalized policy offset in [-1, 1]
                norm_idx = (idx - (n_policies - 1) / 2) / ((n_policies - 1) / 2) if n_policies > 1 else 0.0

                # Enhanced angle jitter for better label distribution
                if angle_deg <= 45 or (135 < angle_deg <= 225) or angle_deg >= 315:
                    angle_jitter = 0.45  # Stronger jitter for top/bottom
                else:
                    angle_jitter = 0.30  # Moderate jitter for left/right
                angle_shifted = angle + norm_idx * angle_jitter

                # Use consistent radius for all labels - professional positioning
                label_r = 1.1  # Fixed radius for all labels

                # Determine text alignment based on angle for optimal readability
                if angle_deg <= 45 or angle_deg >= 315:
                    ha, va = 'center', 'bottom'
                elif 45 < angle_deg <= 135:
                    ha, va = 'left', 'center'
                elif 135 < angle_deg <= 225:
                    ha, va = 'center', 'top'
                else:
                    ha, va = 'right', 'center'

                # Determine label text based on task type with enhanced formatting
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
                    # Show normalized score for other tasks
                    label_text = f"{value:.2f}"

                # Enhanced label styling with professional bbox
                ax.text(angle_shifted, label_r, label_text, ha=ha, va=va, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.26', facecolor='white', edgecolor=color, alpha=0.9, linewidth=1.2),
                        zorder=10, clip_on=False)

    # Enhanced axis customization with professional styling
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
    
    # Enhanced label styling with larger fonts and better positioning
    ax.set_xticklabels(task_labels, fontsize=14, fontweight='bold', ha='center')
    ax.tick_params(axis='x', pad=45)  # Increased padding for professional appearance

    # Enhanced radial axis with consistent styling
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14, alpha=0.95, fontweight='medium')

    # Enhanced radial grid lines with professional appearance
    for tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot([0, 2*pi], [tick, tick], color='gray', alpha=0.22, linewidth=0.8)
    
    # Professional legend with enhanced positioning and styling
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.05, -0.15),
                      borderaxespad=0.0, frameon=True, fancybox=True, shadow=True,
                      fontsize=13, title='ACT Policies', title_fontsize=13)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    legend.get_title().set_fontweight('bold')

    # Enhanced titles with professional positioning
    fig.suptitle('Policy Performance Comparison on Can Sorting Task (Lighting Test)',
                 x=0.23, y=1.0, size=18, fontweight='bold', color='#2c3e50', ha='left')
    fig.text(0.33, 0.975, 'Success Rate by Subtask (0.0 = Failure, 1.0 = Success)', 
             ha='left', va='top', fontsize=14, style='italic', color='#6c757d')

    # Professional layout optimization
    plt.tight_layout(rect=[0.00, 0.00, 1.00, 0.97])
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
    
    # Optimized for A4 format - taller and narrower with better aspect ratio
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), dpi=150)
    axes = axes.flatten()
    
    # Global styling
    fig.patch.set_facecolor('white')
    
    # Create a subplot for each subtask
    for task_idx, task in enumerate(subtasks):
        ax = axes[task_idx]
        row_idx = task_idx // 2
        col_idx = task_idx % 2
        
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
        
        # Create beautiful bars with enhanced styling using consistent colors
        bar_colors = [POLICY_COLOR_MAP.get(p, POLICY_COLORS[i % len(POLICY_COLORS)]) for i,p in enumerate(policies)]
        bars = ax.bar(x, means, yerr=stds, capsize=8,
                      color=bar_colors,
                      alpha=0.85, edgecolor='white', linewidth=2,
                      error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50', 'alpha': 0.8})
        
        # Add gradient effect to bars for professional appearance
        for i, bar in enumerate(bars):
            gradient = plt.Rectangle((bar.get_x(), 0), bar.get_width(), bar.get_height(),
                                   facecolor=bar.get_facecolor(), alpha=0.30, edgecolor='none')
            ax.add_patch(gradient)
        
        # Enhanced subplot styling
        ax.set_facecolor('#fafafa')
        ax.grid(axis='y', linestyle='--', alpha=0.45, linewidth=1, color='#bdc3c7')
        ax.set_axisbelow(True)
        
        # Customize subplot titles with better formatting
        task_title = task.replace(' to ', ' to\n') if len(task) > 20 else task
        ax.set_title(f'{task_title}', fontsize=24, fontweight='bold', 
                    pad=25, color='#2c3e50')
        
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
        
        # Set consistent y-axis limits with professional spacing
        ax.set_ylim(0, 1.42)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        
        # Add a horizontal line indicating 100% success rate
        ax.axhline(y=1.0, color='#27ae60', linestyle='-', alpha=0.7, linewidth=2.5)
        
        # Add value labels on bars with enhanced styling
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            label_y = height + std + 0.05
            
            # Determine label text based on task type with enhanced formatting
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
            
            # Enhanced label styling with professional bbox and colors
            ax.text(bar.get_x() + bar.get_width()/2., label_y, label_text, 
                   ha='center', va='bottom', fontsize=18, fontweight='bold', 
                   color='#2c3e50', rotation=rotation,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=bar_colors[i], alpha=0.92, linewidth=1.4))
        
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
    
    # Professional main title with subtitle - A4 optimized
    fig.suptitle('ACT Policy Performance Analysis: Can Sorting Task (Lighting Test)', 
                fontsize=28, fontweight='bold', y=0.985, color='#2c3e50')
    
    # Add subtitle
    fig.text(0.5, 0.965, 'Mean Success Rate ± Standard Deviation by Subtask', 
             ha='center', va='top', fontsize=20, style='italic', color='#7f8c8d')
    
    # Create combined legend with policies and reference lines
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    
    # Policy legend elements with consistent colors
    policy_legend_elements = []
    for i, policy in enumerate(policies):
        policy_legend_elements.append(
            Rectangle((0, 0), 1, 1, facecolor=POLICY_COLOR_MAP.get(policy, POLICY_COLORS[i % len(POLICY_COLORS)]), 
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
    
    # Position legend across the bottom in multiple rows - closer to plots with bigger text
    legend = fig.legend(handles=all_legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.04),
                       frameon=True, fancybox=True, shadow=True, fontsize=20, 
                       ncol=(len(all_legend_elements) + 2) // 3, columnspacing=1.5, handletextpad=0.8)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_edgecolor('#dee2e6')
    legend.get_frame().set_linewidth(1.4)
    
    # Professional layout with proper spacing optimized for A4 - reduced bottom margin for closer legend
    plt.tight_layout(rect=[0, 0.11, 1, 0.96])
    
    # Save in multiple formats for thesis use
    #plt.savefig(output_dir / 'grouped_bar_plot_with_errors.png', 
    #            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'grouped_bar_plot_with_errors.pdf', 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    #plt.savefig(output_dir / 'grouped_bar_plot_with_errors.svg', 
    #            bbox_inches='tight', facecolor='white', edgecolor='none')
    
    #plt.show()


def create_total_score_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate two analyses: (1) with Return Home, (2) without Return Home.
    Each produces a two-panel figure: overall (Color=all) + color breakdown (red vs green).
    Saves:
        total_score_with_return.pdf
        total_score_without_return.pdf
    """
    def _compute_overall_and_colors(task_label: str, filename: str):
        data = df[df['Task']==task_label]
        if data.empty:
            print(f'No data for {task_label}')
            return
            
        # DEBUG: Check what policies are available in the data
        print(f"\nDEBUG: {task_label} - Available policies: {sorted(data['Policy'].unique())}")
        print(f"DEBUG: Data shape for {task_label}: {data.shape}")
        
        # Professional A4-optimized layout with enhanced styling
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        overall = data[data['Color']=='all'].copy()
        if overall.empty:
            print(f'No overall rows for {task_label}')
        else:
            print(f"DEBUG: Overall data policies for {task_label}: {sorted(overall['Policy'].unique())}")
            print(f"DEBUG: Overall data shape: {overall.shape}")
        
        # Reconstruct component std (exact): recompute components used for that total
        # We need manipulation tasks + (Time [+ReturnHome])
        policies = sorted(overall['Policy'].unique())
        task_is_with = task_label == 'Total Score'
        comp_map = {}
        print(f"DEBUG: Processing {len(policies)} policies for component mapping:")
        for policy in policies:
            print(f"  - Checking policy: {policy}")
            manip_vals = []
            for t in ['Hand Move to Can','Hand Grasp Can','Hand Move to Correct Box','Can in Correct Box']:
                rr=df[(df['Policy']==policy)&(df['Task']==t)&(df['Color']=='all')]
                if rr.empty: 
                    print(f"    Missing task '{t}' for policy {policy}")
                    manip_vals=[]; break
                manip_vals.append(rr['Score'].iloc[0])
                print(f"    Found task '{t}' with score: {rr['Score'].iloc[0]:.3f}")
            if not manip_vals:
                print(f"    SKIP: {policy} - missing manipulation tasks")
                continue
            time_row = df[(df['Policy']==policy)&(df['Task']=='Execution Time')&(df['Color']=='all')]
            if time_row.empty:
                print(f"    SKIP: {policy} - missing Execution Time")
                continue
            time_sc = time_row['Score'].iloc[0]
            print(f"    Found Execution Time with score: {time_sc:.3f}")
            comps = manip_vals + [time_sc]
            if task_is_with:
                rh_row = df[(df['Policy']==policy)&(df['Task']=='Return to Home Position')&(df['Color']=='all')]
                if rh_row.empty:
                    print(f"    SKIP: {policy} - missing Return to Home Position")
                    continue
                rh_sc = rh_row['Score'].iloc[0]
                print(f"    Found Return to Home Position with score: {rh_sc:.3f}")
                comps = manip_vals + [rh_sc, time_sc]
            comp_map[policy] = comps
            print(f"    SUCCESS: {policy} added to component map with {len(comps)} components")
        
        stats_rows=[]
        print(f"DEBUG: Creating stats rows from {len(policies)} policies:")
        for policy in policies:
            row = overall[overall['Policy']==policy]
            if row.empty or policy not in comp_map:
                if row.empty:
                    print(f"  SKIP: {policy} - no overall row found")
                else:
                    print(f"  SKIP: {policy} - not in component map")
                continue
            mean_val = row['Score'].iloc[0]
            std_val = float(np.std(comp_map[policy], ddof=1)) if len(comp_map[policy])>1 else 0.0
            stats_rows.append({'Policy':policy,'Score':mean_val,'Std':std_val})
            print(f"  SUCCESS: {policy} - Score: {mean_val:.3f}, Std: {std_val:.3f}")
        
        print(f"DEBUG: Total stats rows created: {len(stats_rows)}")
        if stats_rows:
            print(f"DEBUG: Final policy list: {[row['Policy'] for row in stats_rows]}")
        
        if not stats_rows:
            print(f'No stats rows for {task_label}')
            return
        
        overall_stats = pd.DataFrame(stats_rows).set_index('Policy')
        
        # Enhanced subplot styling for overall performance
        ax1.set_facecolor('#fafafa')
        ax1.grid(axis='y', linestyle='--', alpha=0.4, linewidth=1, color='#bdc3c7')
        ax1.set_axisbelow(True)
        
        x = np.arange(len(overall_stats.index))
        bars = ax1.bar(x, overall_stats['Score'], yerr=overall_stats['Std'], capsize=10,
                       color=[POLICY_COLOR_MAP.get(p, POLICY_COLORS[i % len(POLICY_COLORS)]) for i,p in enumerate(overall_stats.index)],
                       edgecolor='white', linewidth=2.5, alpha=0.85,
                       error_kw={'elinewidth': 3, 'capthick': 3, 'ecolor': '#34495e', 'alpha': 0.8})
        
        # Professional reference lines
        ax1.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2, zorder=0)
        ax1.axhline(y=0.8, color='#f39c12', linestyle=':', alpha=0.7, linewidth=2, zorder=0)  
        ax1.axhline(y=1.0, color='#27ae60', linestyle='-', alpha=0.8, linewidth=2.5, zorder=0)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels([format_policy_name(p) for p in overall_stats.index], 
                           rotation=45, ha='right', fontsize=12, fontweight='medium', color='#34495e')
        ax1.set_ylabel('Total Score', fontsize=14, fontweight='bold', color='#2c3e50')
        
        # Title adjustments with enhanced styling
        if task_label == 'Total Score Sort Cans(Lighting Test)':
            ax1.set_title('Overall Total Score', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        elif task_label == 'Total Score (No RH)':
            ax1.set_title('Overall Total Score Without RH Subtask (Lighting Test)', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        else:
            ax1.set_title(f'Overall {task_label}', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
        
        ax1.set_ylim(0, 1.2)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.tick_params(axis='y', labelsize=11, colors='#34495e')
        
        # Enhanced styling for subplot borders  
        for spine in ax1.spines.values():
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.5)
        
        # Separate mean and standard deviation labels with enhanced styling
        for i, (bar, val, std, policy) in enumerate(zip(bars, overall_stats['Score'], overall_stats['Std'], overall_stats.index)):
            # Mean value label on bar top with policy color
            policy_color = POLICY_COLOR_MAP.get(policy, POLICY_COLORS[i % len(POLICY_COLORS)])
            ax1.text(bar.get_x() + bar.get_width()/2., val + 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=policy_color, alpha=0.95, linewidth=1.5))
            
            # Standard deviation label on error bar top with neutral styling
            # Skip R-S policy to prevent overlap
            if not (policy.startswith('R-S') and std > 0):
                ax1.text(bar.get_x() + bar.get_width()/2., val + std + 0.06, f'±{std:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='medium',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f9fa', 
                                 edgecolor='#6c757d', alpha=0.9, linewidth=1))
        
        # Enhanced color breakdown subplot
        color_subset = data[data['Color'].isin(['red','green'])]
        if not color_subset.empty:
            ax2.set_facecolor('#fafafa')
            ax2.grid(axis='y', linestyle='--', alpha=0.4, linewidth=1, color='#bdc3c7')
            ax2.set_axisbelow(True)
            
            pivot = color_subset.pivot_table(index='Policy', columns='Color', values='Score', aggfunc='mean')
            pivot.loc['Average'] = pivot.mean(axis=0)
            x2 = np.arange(len(pivot.index))
            width = 0.35
            
            red_vals = pivot['red'] if 'red' in pivot.columns else np.zeros(len(pivot))
            green_vals = pivot['green'] if 'green' in pivot.columns else np.zeros(len(pivot))
            
            # Professional reference lines for second subplot
            ax2.axhline(y=0.5, color='#e74c3c', linestyle='--', alpha=0.6, linewidth=2, zorder=0)
            ax2.axhline(y=0.8, color='#f39c12', linestyle=':', alpha=0.7, linewidth=2, zorder=0)
            ax2.axhline(y=1.0, color='#27ae60', linestyle='-', alpha=0.8, linewidth=2.5, zorder=0)
            
            bars1 = ax2.bar(x2-width/2, red_vals, width, label='Red Cans', color='#e74c3c', 
                           alpha=0.85, edgecolor='white', linewidth=2)
            bars2 = ax2.bar(x2+width/2, green_vals, width, label='Green Cans', color='#2ecc71',
                           alpha=0.85, edgecolor='white', linewidth=2)
            
            # Removed value labels from color breakdown for cleaner presentation
            
            ax2.set_xticks(x2)
            ax2.set_xticklabels([format_policy_name(p) if p!='Average' else 'Average' for p in pivot.index], 
                               rotation=45, ha='right', fontsize=12, fontweight='medium', color='#34495e')
            ax2.set_ylabel('Total Score', fontsize=14, fontweight='bold', color='#2c3e50')
            ax2.set_title(f'{task_label} by Can Color', fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
            ax2.set_ylim(0, 1.2)
            ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax2.tick_params(axis='y', labelsize=11, colors='#34495e')
            
            # Enhanced legend with professional styling
            legend = ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, 
                              shadow=True, framealpha=0.95)
            legend.get_frame().set_facecolor('#f8f9fa')
            legend.get_frame().set_edgecolor('#dee2e6')
            legend.get_frame().set_linewidth(1.5)
            
            # Professional visual separator
            ax2.axvline(len(pivot.index)-1.5, color='#34495e', linestyle='--', alpha=0.6, linewidth=2)
            
            # Enhanced styling for second subplot borders
            for spine in ax2.spines.values():
                spine.set_edgecolor('#bdc3c7')
                spine.set_linewidth(1.5)
        else:
            ax2.set_visible(False)
        
        # Enhanced main titles and reference line legends
        main_title = 'ACT Policy Performance Analysis: Can Sorting Task'
        if task_label == 'Total Score':
            subtitle = 'Total Score Analysis (With Return Home)'
        elif task_label == 'Total Score (No RH)':
            subtitle = 'Total Score Analysis (Without Return Home)'
        else:
            subtitle = f'{task_label} Analysis'
        
        fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.95, color='#2c3e50')
        fig.text(0.5, 0.91, subtitle, ha='center', va='top', fontsize=14, 
                style='italic', color='#7f8c8d')
        
        # Professional multi-row legend with reference lines
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#27ae60', linestyle='-', linewidth=2.5, 
                   label='100% Success', alpha=0.8),
            Line2D([0], [0], color='#f39c12', linestyle=':', linewidth=2, 
                   label='80% Success', alpha=0.7),
            Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2, 
                   label='50% Success', alpha=0.6)
        ]
        
        # Position legend at bottom with 4 columns
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True, 
                  fancybox=True, shadow=True, fontsize=11,
                  title='Success Rate Thresholds', title_fontsize=12)
        
        # Professional layout with proper spacing
        plt.tight_layout(rect=[0, 0.08, 1, 0.88])
        
        plt.savefig(output_dir/filename, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    _compute_overall_and_colors('Total Score','total_score_with_return.pdf')
    _compute_overall_and_colors('Total Score (No RH)','total_score_without_return.pdf')


def create_end_position_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot Return to Home Position success rate per policy (Color='all')."""
    data = df[(df['Task'] == 'Return to Home Position') & (df['Color'] == 'all')]
    if data.empty:
        print('No Return to Home Position data found.')
        return
    stats_df = data.groupby('Policy')['Score'].agg(['mean']).reset_index()
    stats_df = stats_df.sort_values('mean', ascending=False)
    plt.figure(figsize=(10,6), dpi=140)
    colors = plt.get_cmap('tab10')
    x = np.arange(len(stats_df))
    bars = plt.bar(x, stats_df['mean'], color=[POLICY_COLOR_MAP.get(p, POLICY_COLORS[i % len(POLICY_COLORS)]) for i,p in enumerate(stats_df['Policy'])], edgecolor='white', linewidth=1.5)
    plt.xticks(x, [format_policy_name(p) for p in stats_df['Policy']], rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.ylim(0,1.05)
    plt.title('Return to Home Position Success Rate')
    plt.grid(axis='y', alpha=0.3)
    for bar,val in zip(bars, stats_df['mean']):
        plt.text(bar.get_x()+bar.get_width()/2., val+0.02, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'return_home_success.pdf', bbox_inches='tight')
    plt.close()


def create_time_analysis(df: pd.DataFrame, time_info: dict, output_dir: Path) -> None:
    """Plot normalized execution time scores and annotate with real minutes."""
    data = df[(df['Task'] == 'Execution Time') & (df['Color'] == 'all')]
    if data.empty:
        print('No Execution Time data found.')
        return
    stats_df = data.groupby('Policy')['Score'].agg(['mean']).reset_index()
    # For labeling actual minutes
    actual_minutes = []
    for policy in stats_df['Policy']:
        t = time_info.get('policy_times', {}).get(policy, {}).get('minutes', None)
        actual_minutes.append(t)
    stats_df['Minutes'] = actual_minutes
    # Higher score => faster, so sort descending by score
    stats_df = stats_df.sort_values('mean', ascending=False)
    plt.figure(figsize=(10,6), dpi=140)
    colors = plt.get_cmap('tab20')
    x = np.arange(len(stats_df))
    bars = plt.bar(x, stats_df['mean'], color=[POLICY_COLOR_MAP.get(p, POLICY_COLORS[i % len(POLICY_COLORS)]) for i,p in enumerate(stats_df['Policy'])], edgecolor='white', linewidth=1.2)
    plt.xticks(x, [format_policy_name(p) for p in stats_df['Policy']], rotation=45, ha='right')
    plt.ylabel('Normalized Time Score')
    plt.ylim(0,1.05)
    rng = f"({time_info['min_time_minutes']:.1f} - {time_info['max_time_minutes']:.1f} min)"
    plt.title(f'Execution Time Performance {rng}')
    plt.grid(axis='y', alpha=0.3)
    for bar,score,min_val in zip(bars, stats_df['mean'], stats_df['Minutes']):
        label = f"{score:.2f}\n{min_val:.1f}m" if min_val is not None else f"{score:.2f}"
        plt.text(bar.get_x()+bar.get_width()/2., score+0.03, label, ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_analysis.pdf', bbox_inches='tight')
    plt.close()


def create_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save summary statistics including both total score variants if present."""
    # Overall Performance by Total Score (with RH) if present, else fallback to other
    lines = []
    has_with = 'Total Score' in df['Task'].unique()
    has_without = 'Total Score (No RH)' in df['Task'].unique()
    if has_with:
        overall_with = df[(df['Task']=='Total Score')&(df['Color']=='all')].groupby('Policy')['Score'].agg(['mean','std'])
    else:
        overall_with = pd.DataFrame()
    if has_without:
        overall_without = df[(df['Task']=='Total Score (No RH)')&(df['Color']=='all')].groupby('Policy')['Score'].agg(['mean','std'])
    else:
        overall_without = pd.DataFrame()
    task_stats = df[df['Color']=='all'].groupby(['Policy','Task'])['Score'].agg(['count','mean','std']).round(3)
    color_stats = df.groupby(['Policy','Color'])['Score'].agg(['count','mean','std']).round(3)
    with open(output_dir/'summary_statistics.txt','w') as f:
        f.write('CAN MANIPULATION POLICY ANALYSIS REPORT\n' + '='*55 + '\n\n')
        if has_with:
            f.write('1. OVERALL TOTAL SCORE (With Return Home)\n'+'-'*45+'\n')
            f.write(overall_with.round(3).to_string())
            f.write('\n\n')
        if has_without:
            f.write('2. OVERALL TOTAL SCORE (Without Return Home)\n'+'-'*48+'\n')
            f.write(overall_without.round(3).to_string())
            f.write('\n\n')
        f.write('3. PERFORMANCE BY POLICY AND TASK (Color=all)\n'+'-'*45+'\n')
        f.write(task_stats.to_string()); f.write('\n\n')
        f.write('4. PERFORMANCE BY POLICY AND COLOR (raw aggregates)\n'+'-'*50+'\n')
        f.write(color_stats.to_string()); f.write('\n')
    print(f"Summary statistics saved to: {output_dir / 'summary_statistics.txt'}")
    if has_with:
        print('\nOverall (With RH):')
        print(overall_with['mean'].sort_values(ascending=False))
    if has_without:
        print('\nOverall (No RH):')
        print(overall_without['mean'].sort_values(ascending=False))


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize ACT policy performance on can manipulation tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path", "-c",
        type=str,
        default="plot_wandb/can_policies_time.csv",
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
        # Build global color map once policies known
        global POLICY_COLOR_MAP
        POLICY_COLOR_MAP = {p: POLICY_COLORS[i % len(POLICY_COLORS)] for i, p in enumerate(sorted(df['Policy'].unique()))}
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
        selected_plots = ["radar", "bars", "summary", "end_position", "time_analysis", "total_score"]
    
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
    if "total_score" in selected_plots:
        create_total_score_analysis(df, output_dir)
    # ...existing code...
    print(f"\nAnalysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit(main())
