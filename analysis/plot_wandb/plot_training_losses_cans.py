import argparse
from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up the plotting style for thesis-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Configure matplotlib for high-quality output
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "text.usetex": False,  # LaTeX formatting handled manually
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# Policy-specific colors for can manipulation policies
POLICY_COLORS = [
    '#3498db',  # Blue R-A    
    '#4b0082',  # Indigo R-A-AUG 
    '#228b22',  # Forest Green R-A-P
    '#dc143c',  # Crimson R-SW 
    '#ffd700',  # Gold  R-S_LWA
    '#9b59b6',  # Purple R-WA
    '#1abc9c',  # Teal R-WA-P
    '#34495e',  # Slate Gray R-WA-PV_AT_A
    '#ff6347',  # Tomato R-W_RA
]




def format_policy_name_with_subscripts(name):
    """
    Format policy name to display subscripts correctly using matplotlib formatting.
    Converts underscore notation so only the next letter after underscore becomes subscript.
    Examples: 'S_LWA' -> 'S$_L$WA', 'R-S_LWA' -> 'R-S$_L$WA', 'A_B_C' -> 'A$_B$$_C$'
    """
    result = ""
    i = 0
    
    while i < len(name):
        if name[i] == '_' and i + 1 < len(name):
            # Found underscore with character after it
            subscript_char = name[i + 1]
            result += f"$_{{{subscript_char}}}$"
            i += 2  # Skip both underscore and the subscript character
        else:
            # Regular character, add it to result
            result += name[i]
            i += 1
    
    return result


# Complete the policy color map
POLICY_COLOR_MAP = {
    'R-A': '#3498db',
    'R-A-AUG': '#4b0082', 
    'R-A-P': '#228b22',
    'R-SW': '#dc143c',
    'R-S_LWA': '#ffd700',
    'R-WA': '#9b59b6',
    'R-WA-P': '#1abc9c',
    'R-WA-PV_AT_A': '#34495e',
    'R-W_RA': '#ff6347',
}

def load_and_clean_data(csv_path):
    """Load CSV data and clean it for plotting."""
    df = pd.read_csv(csv_path)

    # Clean column names and extract model names
    models = {}
    for col in df.columns:
        if "train/l1_loss" in col and not ("MIN" in col or "MAX" in col):
            model_name = col.split(" - ")[0]
            models[model_name] = col

    return df, models


def create_training_loss_plot(df, models, output_dir):
    """Create a comprehensive training loss plot."""
    fig, ax = plt.subplots(figsize=(14, 9))  # Slightly larger figure

    for i, (model_name, col_name) in enumerate(models.items()):
        # Get data and remove NaN values
        data = df[["Step", col_name]].dropna()

        # Convert step to thousands for better readability
        steps_k = data["Step"] / 1000
        losses = data[col_name]

        # Get color for this policy
        color = POLICY_COLOR_MAP.get(model_name, POLICY_COLORS[i % len(POLICY_COLORS)])

        # Plot the main line
        ax.plot(
            steps_k,
            losses,
            label=format_policy_name_with_subscripts(model_name),
            linewidth=3.0,  # Thicker lines
            color=color,
            alpha=0.8,
        )

        # Add a smoothed trend line
        if len(data) > 10:
            window_size = max(5, len(data) // 20)
            smoothed = losses.rolling(window=window_size, center=True).mean()
            ax.plot(
                steps_k,
                smoothed,
                "--",
                color=color,
                alpha=0.6,
                linewidth=2.0,  # Thicker smoothed lines
            )

    ax.set_xlabel("Training Steps (×1000)", fontweight="bold", fontsize=16)  # Larger font
    ax.set_ylabel("L1 Loss", fontweight="bold", fontsize=16)  # Larger font
    ax.set_title("Can Policies Training Loss Comparison", fontweight="bold", pad=25, fontsize=20)  # Larger title

    # Improve the legend with larger font
    ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right", fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Set x-axis to start from 0
    ax.set_xlim(left=0)
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)

    # Add some styling with larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    # Save in both formats
    plt.savefig(output_dir / "can_policies_training_comparison.png", format="png")
    plt.savefig(output_dir / "can_policies_training_comparison.pdf", format="pdf")



def create_individual_model_plots(df, models, output_dir, ncols: int = 2):  # Changed default to 2
    """Create all individual model plots as subplots in a grid with 2 columns."""
    model_items = list(models.items())
    n_models = len(model_items)
    if n_models == 0:
        return

    ncols = 2  # Force 2 columns as requested
    nrows = int(ceil(n_models / ncols))

    # Smaller width for tighter spacing, same height for readability
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 4.5 * nrows), squeeze=False
    )

    for idx, (model_name, col_name) in enumerate(model_items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # Data
        data = df[["Step", col_name]].dropna()
        if len(data) == 0:
            ax.set_visible(False)
            continue

        steps_k = data["Step"] / 1000
        losses = data[col_name]

        # Get color for this policy
        color = POLICY_COLOR_MAP.get(model_name, POLICY_COLORS[idx % len(POLICY_COLORS)])

        # Main series with policy-specific color
        ax.plot(steps_k, losses, linewidth=2.2, color=color, alpha=0.9)
        ax.fill_between(steps_k, losses, alpha=0.20, color=color)

        # Trend line (smoothed)
        if len(data) > 10:
            window_size = max(5, len(data) // 15)
            smoothed = losses.rolling(window=window_size, center=True).mean()
            ax.plot(steps_k, smoothed, "--", color="#ff5100", alpha=0.8, linewidth=2.0)

        # Improvement annotation (larger text)
        if len(losses) > 1:
            init = losses.iloc[0]
            final = losses.iloc[-1]
            if init != 0:
                improvement = ((init - final) / init) * 100
                ax.text(
                    0.05,
                    0.98,
                    f"Δ {improvement:.1f}%",
                    transform=ax.transAxes,
                    fontsize=16,  # Increased from 14
                    fontweight="bold",
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                )

        # Only show Y-axis label and ticks for leftmost plots
        if c == 0:  # Left column
            ax.set_ylabel("L1 Loss", fontsize=20, fontweight="bold")  # Increased from 18
            ax.tick_params(axis='y', labelsize=18)  # Increased from 16
        else:  # Right column - hide y-axis labels but keep ticks
            ax.tick_params(axis='y', labelleft=False, labelsize=18)  # Increased from 16

        ax.set_xlabel("Steps (×1000)", fontsize=18)  # Increased from 16
        ax.set_title(format_policy_name_with_subscripts(model_name), 
                    fontsize=20, pad=12, fontweight="bold")  # Increased from 18
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='x', labelsize=18)  # Increased from 16
        
        # Style spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add visual separation between subplots in the same row
        col_idx = c  # Use consistent naming
        if col_idx == 0:  # Left subplot - add right border
            ax.spines['right'].set_visible(True)
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

    # Hide any unused axes
    for idx in range(n_models, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97], w_pad=1.0, h_pad=2.0)  # Tighter horizontal spacing
    plt.savefig(output_dir / "can_policies_individual_grid.png", format="png")
    plt.savefig(output_dir / "can_policies_individual_grid.pdf", format="pdf")



def create_convergence_analysis_plot(df, models, output_dir):
    """Create a plot showing convergence behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Log scale for better visualization of convergence
    for i, (model_name, col_name) in enumerate(models.items()):
        data = df[["Step", col_name]].dropna()
        if len(data) == 0:
            continue

        steps_k = data["Step"] / 1000
        losses = data[col_name]

        # Get policy-specific color
        color = POLICY_COLOR_MAP.get(model_name, POLICY_COLORS[i % len(POLICY_COLORS)])

        ax1.semilogy(
            steps_k,
            losses,
            label=format_policy_name_with_subscripts(model_name),
            linewidth=2.5,
            color=color,
            alpha=0.8,
        )

    ax1.set_xlabel("Training Steps (×1000)", fontweight="bold", fontsize=14)  # Larger font
    ax1.set_ylabel("L1 Loss (log scale)", fontweight="bold", fontsize=14)  # Larger font
    ax1.set_title("Training Loss Convergence (Log Scale)", fontweight="bold", fontsize=16)  # Larger title
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)  # Larger legend
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)  # Larger axis labels

    # Plot 2: Learning rate analysis (loss reduction per 1000 steps)
    for i, (model_name, col_name) in enumerate(models.items()):
        data = df[["Step", col_name]].dropna()
        if len(data) < 2:
            continue

        steps = data["Step"].values
        losses = data[col_name].values

        # Calculate loss reduction rate
        loss_diff = np.diff(losses)
        step_diff = np.diff(steps)
        loss_rate = loss_diff / step_diff * 1000  # Per 1000 steps

        # Get policy-specific color
        color = POLICY_COLOR_MAP.get(model_name, POLICY_COLORS[i % len(POLICY_COLORS)])

        ax2.plot(
            steps[1:] / 1000,
            np.abs(loss_rate),
            label=format_policy_name_with_subscripts(model_name),
            linewidth=2.5,
            color=color,
            alpha=0.8,
        )

    ax2.set_xlabel("Training Steps (×1000)", fontweight="bold", fontsize=14)  # Larger font
    ax2.set_ylabel("|Loss Reduction Rate| (per 1K steps)", fontweight="bold", fontsize=14)  # Larger font
    ax2.set_title("Learning Rate Analysis", fontweight="bold", fontsize=16)  # Larger title
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)  # Larger legend
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax2.tick_params(axis='both', labelsize=12)  # Larger axis labels

    plt.tight_layout()
    plt.savefig(output_dir / "can_policies_convergence_analysis.png", format="png")
    plt.savefig(output_dir / "can_policies_convergence_analysis.pdf", format="pdf")



def create_summary_statistics_plot(df, models, output_dir):
    """Create a summary statistics visualization."""
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))  # Single plot, larger size

    model_names = []
    final_losses = []
    original_names = []  # Keep track of original names for color mapping

    for model_name, col_name in models.items():
        data = df[col_name].dropna()
        if len(data) == 0:
            continue

        model_names.append(format_policy_name_with_subscripts(model_name))
        final_losses.append(data.iloc[-1])
        original_names.append(model_name)  # Store original name

    # Sort by final loss for better visualization
    sorted_data = sorted(zip(final_losses, model_names, original_names))
    sorted_losses, sorted_names, sorted_originals = zip(*sorted_data) if sorted_data else ([], [], [])

    # Use policy-specific colors for each bar using original names
    bar_colors = [POLICY_COLOR_MAP.get(orig_name, POLICY_COLORS[i % len(POLICY_COLORS)]) 
                 for i, orig_name in enumerate(sorted_originals)]

    # Bar plot of final losses
    bars1 = ax1.bar(
        range(len(sorted_names)),
        sorted_losses,
        alpha=0.8,
        color=bar_colors,
        edgecolor="none",  # Remove blue border
        linewidth=0,
    )
    ax1.set_ylabel("Final L1 Loss", fontweight="bold", fontsize=16)  # Larger font
    ax1.set_title("Final Training Loss by Policy", fontweight="bold", fontsize=18)  # Larger title
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, fontsize=14)  # Larger labels
    ax1.tick_params(axis='y', labelsize=14)  # Larger y-axis labels
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 0.07)  # Increase y-axis range

    # Add value labels on bars with colored boxes
    for i, (bar, value) in enumerate(zip(bars1, sorted_losses)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + height * 0.02,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,  # Larger value labels
            fontweight="bold",
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=bar_colors[i], alpha=0.92, linewidth=1.4)
        )

    plt.tight_layout()
    plt.savefig(output_dir / "can_policies_final_loss_summary.png", format="png")
    plt.savefig(output_dir / "can_policies_final_loss_summary.pdf", format="pdf")



def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate beautiful training loss plots from WandB CSV export data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_training_losses.py
  python plot_training_losses.py --csv_path my_data.csv
  python plot_training_losses.py --output_dir my_plots
  python plot_training_losses.py --plots comparison individual
  python plot_training_losses.py --csv_path my_data.csv --output_dir my_plots --plots all
        """,
    )

    parser.add_argument(
        "--csv_path",
        "-c",
        type=str,
        default="wandb/wandb_export_2025-08-13T15_25_06.121+02_00.csv",
        help="Path to the CSV file containing training data (default: wandb/wandb_export_2025-08-13T15_25_06.121+02_00.csv)",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="plot_wandb/plots",
        help="Directory to save output plots (default: plot_wandb/plots)",
    )

    parser.add_argument(
        "--plots",
        "-p",
        nargs="*",
        choices=["comparison", "individual", "convergence", "summary", "all"],
        default=["all"],
        help="Choose which plots to create. Options: comparison, individual, convergence, summary, all (default: all)",
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Do not display plots interactively (useful for headless environments)",
    )

    parser.add_argument(
        "--max_steps",
        "-s",
        type=int,
        default=None,
        help="Only use data points with Step <= this value (e.g., 80000)",
    )
    
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns for the individual-plots grid (default: 3)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    # Handle plot selection
    if "all" in args.plots:
        selected_plots = ["comparison", "individual", "convergence", "summary"]
    else:
        selected_plots = args.plots

    # Validate input file exists
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' not found!")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Selected plots: {', '.join(selected_plots)}")
    print()

    print("Loading and processing data...")
    try:
        df, models = load_and_clean_data(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return 1
    
    if args.max_steps is not None:
        if "Step" not in df.columns:
            print("Error: 'Step' column not found in the CSV.")
            return 1
        before = len(df)
        df = df[df["Step"] <= args.max_steps].copy()
        if df.empty:
            print(f"Error: No rows with Step <= {args.max_steps}.")
            return 1
        print(f"Applied step limit: Step <= {args.max_steps} "
              f"(rows kept: {len(df)}/{before})")

    if not models:
        print("No training loss data found in the CSV file!")
        return 1

    print(f"Found {len(models)} models: {list(models.keys())}")
    print()

    # Set matplotlib backend for headless environments
    if args.no_display:
        import matplotlib

        matplotlib.use("Agg")
        # Disable plt.show() calls by monkey patching
        plt.show = lambda: None

    # Create selected plots
    if "comparison" in selected_plots:
        print("Creating training loss comparison plot...")
        create_training_loss_plot(df, models, output_dir)

    if "individual" in selected_plots:
        print("Creating individual model plots grid...")
        create_individual_model_plots(df, models, output_dir, ncols=args.grid_cols)

    if "convergence" in selected_plots:
        print("Creating convergence analysis plot...")
        create_convergence_analysis_plot(df, models, output_dir)

    if "summary" in selected_plots:
        print("Creating summary statistics plot...")
        create_summary_statistics_plot(df, models, output_dir)

    print()
    print(f"✓ Selected plots saved to {output_dir}/ in PNG formats")
    return 0


if __name__ == "__main__":
    exit(main())
