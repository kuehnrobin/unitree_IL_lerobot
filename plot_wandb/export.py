"""
Export utilities for converting plots to various formats suitable for academic publications.

This module provides functionality to export plots as SVG, PDF, PNG, and even generate
LaTeX code for direct inclusion in academic papers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import subprocess
import os


class ExportManager:
    """
    Manager for exporting plots in various formats suitable for publications.
    """
    
    def __init__(self, output_dir: Union[str, Path] = "plots"):
        """
        Initialize the export manager.
        
        Args:
            output_dir: Base directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_figure(self, 
                     fig: plt.Figure, 
                     filename: str,
                     formats: List[str] = ["png", "svg", "pdf"],
                     dpi: int = 300,
                     bbox_inches: str = "tight",
                     transparent: bool = False) -> List[Path]:
        """
        Export figure in multiple formats.
        
        Args:
            fig: matplotlib Figure to export
            filename: Base filename (without extension)
            formats: List of formats to export
            dpi: DPI for raster formats
            bbox_inches: Bounding box mode
            transparent: Whether to use transparent background
            
        Returns:
            List of paths to exported files
        """
        exported_files = []
        
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            
            # Format-specific settings
            save_kwargs = {
                "format": fmt,
                "bbox_inches": bbox_inches,
                "transparent": transparent
            }
            
            if fmt in ["png", "jpg", "jpeg", "tiff"]:
                save_kwargs["dpi"] = dpi
            
            # Special handling for SVG to ensure text is editable
            if fmt == "svg":
                save_kwargs["transparent"] = True
                
            fig.savefig(filepath, **save_kwargs)
            exported_files.append(filepath)
            print(f"Exported {fmt.upper()} to {filepath}")
            
        return exported_files
    
    def generate_latex_figure(self, 
                            filename: str,
                            caption: str,
                            label: str,
                            width: str = "0.8\\textwidth",
                            placement: str = "htbp") -> str:
        """
        Generate LaTeX code for including a figure.
        
        Args:
            filename: Filename of the figure (without path)
            caption: Figure caption
            label: Figure label for referencing
            width: Figure width in LaTeX units
            placement: Figure placement options
            
        Returns:
            LaTeX code string
        """
        latex_code = f"""\\begin{{figure}}[{placement}]
    \\centering
    \\includegraphics[width={width}]{{{filename}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}"""
        
        return latex_code
    
    def generate_subfigure_latex(self, 
                                figures: List[Dict[str, str]], 
                                main_caption: str,
                                main_label: str,
                                placement: str = "htbp") -> str:
        """
        Generate LaTeX code for subfigures.
        
        Args:
            figures: List of dicts with 'filename', 'caption', 'label', 'width'
            main_caption: Main figure caption
            main_label: Main figure label
            placement: Figure placement options
            
        Returns:
            LaTeX code string
        """
        latex_code = f"\\begin{{figure}}[{placement}]\n    \\centering\n"
        
        for fig_info in figures:
            width = fig_info.get('width', '0.45\\textwidth')
            latex_code += f"""    \\begin{{subfigure}}{{{width}}}
        \\centering
        \\includegraphics[width=\\textwidth]{{{fig_info['filename']}}}
        \\caption{{{fig_info['caption']}}}
        \\label{{fig:{fig_info['label']}}}
    \\end{{subfigure}}%
"""
            if figures.index(fig_info) < len(figures) - 1:
                latex_code += "    \\hfill\n"
        
        latex_code += f"""    \\caption{{{main_caption}}}
    \\label{{fig:{main_label}}}
\\end{{figure}}"""
        
        return latex_code
    
    def save_latex_code(self, latex_code: str, filename: str):
        """
        Save LaTeX code to a file.
        
        Args:
            latex_code: LaTeX code string
            filename: Output filename
        """
        filepath = self.output_dir / f"{filename}.tex"
        with open(filepath, 'w') as f:
            f.write(latex_code)
        print(f"Saved LaTeX code to {filepath}")
    
    def create_thesis_plot_package(self, 
                                  figures: List[plt.Figure],
                                  figure_names: List[str],
                                  captions: List[str],
                                  labels: List[str],
                                  main_title: str = "Training Results") -> Dict[str, List[Path]]:
        """
        Create a complete package of plots for thesis with all formats and LaTeX code.
        
        Args:
            figures: List of matplotlib figures
            figure_names: List of base filenames
            captions: List of figure captions
            labels: List of figure labels
            main_title: Main title for the figure package
            
        Returns:
            Dictionary with exported file paths by format
        """
        if not (len(figures) == len(figure_names) == len(captions) == len(labels)):
            raise ValueError("All input lists must have the same length")
        
        exported_files = {"png": [], "svg": [], "pdf": [], "latex": []}
        
        # Export all figures
        for fig, name, caption, label in zip(figures, figure_names, captions, labels):
            files = self.export_figure(fig, name, ["png", "svg", "pdf"])
            
            for file_path in files:
                format_key = file_path.suffix[1:]  # Remove the dot
                exported_files[format_key].append(file_path)
            
            # Generate individual LaTeX code
            latex_code = self.generate_latex_figure(
                f"{name}.pdf", caption, label
            )
            latex_file = self.output_dir / f"{name}.tex"
            with open(latex_file, 'w') as f:
                f.write(latex_code)
            exported_files["latex"].append(latex_file)
        
        # Generate combined LaTeX file
        combined_latex = self._generate_combined_latex(figure_names, captions, labels, main_title)
        combined_file = self.output_dir / "all_figures.tex"
        with open(combined_file, 'w') as f:
            f.write(combined_latex)
        exported_files["latex"].append(combined_file)
        
        # Create a README file
        self._create_package_readme(exported_files)
        
        return exported_files
    
    def _generate_combined_latex(self, 
                               figure_names: List[str],
                               captions: List[str], 
                               labels: List[str],
                               main_title: str) -> str:
        """Generate LaTeX code that includes all figures."""
        latex_code = f"% {main_title} - Generated Figure Package\n"
        latex_code += "% Include these figures in your thesis\n\n"
        
        for name, caption, label in zip(figure_names, captions, labels):
            latex_code += self.generate_latex_figure(f"{name}.pdf", caption, label)
            latex_code += "\n\n"
        
        # Add reference section
        latex_code += "% References to figures:\n"
        for label in labels:
            latex_code += f"% \\ref{{fig:{label}}}\n"
        
        return latex_code
    
    def _create_package_readme(self, exported_files: Dict[str, List[Path]]):
        """Create a README file for the exported package."""
        readme_content = """# Plot Export Package

This directory contains plots exported for thesis/publication use.

## File Structure:

### PNG Files (High Resolution)
- Use for presentations or online viewing
- 300 DPI resolution for high quality

### SVG Files (Vector Graphics)
- Use for web or when you need scalable graphics
- Text remains editable in vector graphics software

### PDF Files (Publication Ready)
- Use for LaTeX documents and print publications
- Recommended for thesis inclusion
- Vector format ensures crisp printing at any size

### LaTeX Files
- Ready-to-use LaTeX code for including figures
- Individual .tex files for each figure
- all_figures.tex contains all figures combined

## Usage in LaTeX:

1. Copy the PDF files to your thesis figures directory
2. Include the LaTeX code from the .tex files
3. Make sure to include these packages in your preamble:
   ```latex
   \\usepackage{graphicx}
   \\usepackage{subcaption}  % if using subfigures
   ```

## Figure Quality:
- All figures exported at 300 DPI
- Vector formats (PDF/SVG) scale without quality loss
- Consistent styling optimized for academic publications
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print(f"Created README at {readme_file}")
    
    def optimize_pdfs(self, pdf_files: List[Path]) -> List[Path]:
        """
        Optimize PDF files for smaller size while maintaining quality.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of optimized PDF file paths
        """
        optimized_files = []
        
        for pdf_file in pdf_files:
            try:
                optimized_file = pdf_file.parent / f"{pdf_file.stem}_optimized.pdf"
                
                # Use ghostscript for optimization if available
                cmd = [
                    "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4",
                    "-dPDFSETTINGS=/printer", "-dNOPAUSE", "-dQUIET", "-dBATCH",
                    f"-sOutputFile={optimized_file}", str(pdf_file)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    optimized_files.append(optimized_file)
                    print(f"Optimized {pdf_file.name} -> {optimized_file.name}")
                else:
                    print(f"Failed to optimize {pdf_file.name}, keeping original")
                    optimized_files.append(pdf_file)
                    
            except FileNotFoundError:
                print("Ghostscript not found, skipping PDF optimization")
                optimized_files.append(pdf_file)
            except Exception as e:
                print(f"Error optimizing {pdf_file.name}: {e}")
                optimized_files.append(pdf_file)
        
        return optimized_files
    
    def create_plot_grid(self, 
                        figures: List[plt.Figure],
                        grid_shape: Tuple[int, int],
                        figure_size: Tuple[float, float] = (16, 12),
                        title: str = "Training Results Overview") -> plt.Figure:
        """
        Create a grid of multiple plots in a single figure.
        
        Args:
            figures: List of matplotlib figures to combine
            grid_shape: (rows, cols) for the grid
            figure_size: Size of the combined figure
            title: Title for the combined figure
            
        Returns:
            Combined matplotlib figure
        """
        rows, cols = grid_shape
        fig = plt.figure(figsize=figure_size)
        fig.suptitle(title, fontsize=16, y=0.98)
        
        for i, source_fig in enumerate(figures[:rows*cols]):
            if i >= rows * cols:
                break
                
            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Copy content from source figure
            source_ax = source_fig.axes[0] if source_fig.axes else None
            if source_ax:
                # Copy lines
                for line in source_ax.get_lines():
                    ax.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), 
                           label=line.get_label(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth())
                
                # Copy labels and title
                ax.set_xlabel(source_ax.get_xlabel())
                ax.set_ylabel(source_ax.get_ylabel())
                ax.set_title(source_ax.get_title())
                
                # Copy legend if it exists
                if source_ax.get_legend():
                    ax.legend()
                
                # Copy grid
                ax.grid(source_ax.xaxis._gridOnMajor)
        
        plt.tight_layout()
        return fig
