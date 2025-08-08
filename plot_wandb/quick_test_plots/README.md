# Plot Export Package

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
   \usepackage{graphicx}
   \usepackage{subcaption}  % if using subfigures
   ```

## Figure Quality:
- All figures exported at 300 DPI
- Vector formats (PDF/SVG) scale without quality loss
- Consistent styling optimized for academic publications
