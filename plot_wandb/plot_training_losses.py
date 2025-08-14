import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up the plotting style for thesis-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for high-quality output
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_and_clean_data(csv_path):
    """Load CSV data and clean it for plotting."""
    df = pd.read_csv(csv_path)
    
    # Clean column names and extract model names
    models = {}
    for col in df.columns:
        if 'train/l1_loss' in col and not ('MIN' in col or 'MAX' in col):
            model_name = col.split(' - ')[0]
            models[model_name] = col
    
    return df, models

def create_training_loss_plot(df, models, output_dir):
    """Create a comprehensive training loss plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model_name, col_name) in enumerate(models.items()):
        # Get data and remove NaN values
        data = df[['Step', col_name]].dropna()
        
        # Convert step to thousands for better readability
        steps_k = data['Step'] / 1000
        losses = data[col_name]
        
        # Plot the main line
        ax.plot(steps_k, losses, label=model_name.replace('_', ' ').title(), 
                linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
        
        # Add a smoothed trend line
        if len(data) > 10:
            window_size = max(5, len(data) // 20)
            smoothed = losses.rolling(window=window_size, center=True).mean()
            ax.plot(steps_k, smoothed, '--', color=colors[i % len(colors)], 
                   alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Training Steps (×1000)', fontweight='bold')
    ax.set_ylabel('L1 Loss', fontweight='bold')
    ax.set_title('Training Loss Comparison Across Models', fontweight='bold', pad=20)
    
    # Improve the legend
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis to start from 0
    ax.set_xlim(left=0)
    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(bottom=0)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(output_dir / 'training_loss_comparison.png', format='png')
    plt.savefig(output_dir / 'training_loss_comparison.svg', format='svg')
    plt.show()

def create_individual_model_plots(df, models, output_dir):
    """Create individual plots for each model."""
    for model_name, col_name in models.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data and remove NaN values
        data = df[['Step', col_name]].dropna()
        
        if len(data) == 0:
            continue
            
        steps_k = data['Step'] / 1000
        losses = data[col_name]
        
        # Plot with filled area under curve
        ax.plot(steps_k, losses, linewidth=2.5, color='#1f77b4', alpha=0.8)
        ax.fill_between(steps_k, losses, alpha=0.3, color='#1f77b4')
        
        # Add trend line
        if len(data) > 10:
            window_size = max(5, len(data) // 15)
            smoothed = losses.rolling(window=window_size, center=True).mean()
            ax.plot(steps_k, smoothed, '--', color='#ff7f0e', 
                   alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
        
        # Calculate and display final loss improvement
        if len(losses) > 1:
            initial_loss = losses.iloc[0]
            final_loss = losses.iloc[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            ax.text(0.10, 0.98, f'Improvement: {improvement:.1f}%', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Training Steps (×1000)', fontweight='bold')
        ax.set_ylabel('L1 Loss', fontweight='bold')
        ax.set_title(f'{model_name.replace("_", " ").title()} - Training Loss', 
                    fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual plots
        safe_name = model_name.replace(' ', '_').replace('-', '_')
        plt.savefig(output_dir / f'{safe_name}_training_loss.png', format='png')
        plt.savefig(output_dir / f'{safe_name}_training_loss.svg', format='svg')
        plt.show()

def create_convergence_analysis_plot(df, models, output_dir):
    """Create a plot showing convergence behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Log scale for better visualization of convergence
    for i, (model_name, col_name) in enumerate(models.items()):
        data = df[['Step', col_name]].dropna()
        if len(data) == 0:
            continue
            
        steps_k = data['Step'] / 1000
        losses = data[col_name]
        
        ax1.semilogy(steps_k, losses, label=model_name.replace('_', ' ').title(), 
                    linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Training Steps (×1000)', fontweight='bold')
    ax1.set_ylabel('L1 Loss (log scale)', fontweight='bold')
    ax1.set_title('Training Loss Convergence (Log Scale)', fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate analysis (loss reduction per 1000 steps)
    for i, (model_name, col_name) in enumerate(models.items()):
        data = df[['Step', col_name]].dropna()
        if len(data) < 2:
            continue
            
        steps = data['Step'].values
        losses = data[col_name].values
        
        # Calculate loss reduction rate
        loss_diff = np.diff(losses)
        step_diff = np.diff(steps)
        loss_rate = loss_diff / step_diff * 1000  # Per 1000 steps
        
        ax2.plot(steps[1:] / 1000, np.abs(loss_rate), 
                label=model_name.replace('_', ' ').title(), 
                linewidth=2.5, color=colors[i % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('Training Steps (×1000)', fontweight='bold')
    ax2.set_ylabel('|Loss Reduction Rate| (per 1K steps)', fontweight='bold')
    ax2.set_title('Learning Rate Analysis', fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', format='png')
    plt.savefig(output_dir / 'convergence_analysis.svg', format='svg')
    plt.show()

def create_summary_statistics_plot(df, models, output_dir):
    """Create a summary statistics visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = []
    final_losses = []
    initial_losses = []
    min_losses = []
    
    for model_name, col_name in models.items():
        data = df[col_name].dropna()
        if len(data) == 0:
            continue
            
        model_names.append(model_name.replace('_', ' ').title())
        initial_losses.append(data.iloc[0])
        final_losses.append(data.iloc[-1])
        min_losses.append(data.min())
    
    # Bar plot of final losses
    bars1 = ax1.bar(model_names, final_losses, alpha=0.7, color='skyblue', 
                   edgecolor='navy', linewidth=1.5)
    ax1.set_ylabel('Final L1 Loss', fontweight='bold')
    ax1.set_title('Final Training Loss by Model', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, final_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement percentage plot
    improvements = [((init - final) / init * 100) for init, final in zip(initial_losses, final_losses)]
    bars2 = ax2.bar(model_names, improvements, alpha=0.7, color='lightgreen', 
                   edgecolor='darkgreen', linewidth=1.5)
    ax2.set_ylabel('Loss Improvement (%)', fontweight='bold')
    ax2.set_title('Training Loss Improvement', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_statistics.png', format='png')
    plt.savefig(output_dir / 'summary_statistics.svg', format='svg')
    plt.show()

def main():
    # Set up paths
    csv_path = Path('wandb/wandb_export_2025-08-13T15_25_06.121+02_00.csv')
    output_dir = Path('plot_wandb/plots')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading and processing data...")
    df, models = load_and_clean_data(csv_path)
    
    print(f"Found {len(models)} models: {list(models.keys())}")
    
    print("Creating training loss comparison plot...")
    create_training_loss_plot(df, models, output_dir)
    
    print("Creating individual model plots...")
    create_individual_model_plots(df, models, output_dir)
    
    print("Creating convergence analysis plot...")
    create_convergence_analysis_plot(df, models, output_dir)
    
    print("Creating summary statistics plot...")
    create_summary_statistics_plot(df, models, output_dir)
    
    print(f"All plots saved to {output_dir}/ in both PNG and SVG formats")
    print("PNG files are recommended for insertion in documents")
    print("SVG files are vector graphics, perfect for presentations and high-quality printing")

if __name__ == "__main__":
    main()