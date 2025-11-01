# MPCO: Meta-Prompting for Code Optimization

**Official repository for the ASE'25 paper: "Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective"**

## Overview

MPCO addresses the cross-model prompt engineering bottleneck in industrial LLM-based code optimization platforms by automatically generating model-specific optimization prompts. This repository contains the evaluation results and analysis scripts from our comprehensive study across 5 real-world systems and 3 major LLMs.

### Key Results

- **RQ1**: MPCO consistently outperforms baseline prompting methods with average rank 1.00 across all systems
- **RQ2**: Comprehensive context integration is essential - removing any component significantly degrades performance
- **RQ3**: All three major LLMs can serve effectively as meta-prompters


### Subject Systems

- **BitmapPlusPlus**: High-performance bitmap processing (C++)
- **Llama.cpp**: Efficient LLM inference engine (C++)  
- **RPCS3**: PlayStation 3 emulator (C++)
- **Faster-Whisper**: Optimized speech recognition (Python)
- **Langflow**: Visual programming for language models (Python)

## Repository Structure

```
MPCO/
├── analysis/              # Analysis scripts for generating paper results
└── results/               # Evaluation data and generated tables
```


## Analysis Scripts

1. **`generate_RQs_latex_table.py`**: Main script that generates all RQ result tables by:
   - Loading evaluation data from CSV files
   - Calculating %PI (Percentage Performance Improvement) 
   - Performing statistical tests (Mann-Whitney U test and Cohen's d)
   - Ranking approaches and generating LaTeX tables

2. **`plot_cross_model_challenge.py`**: Generates cross-model analysis showing how prompts optimized for one LLM perform when used with other LLMs

3. **`generate_intro_latex_table.py`**: Creates introduction tables showing system-specific results

4. **`discussion_analysis.py`**: Generates analysis results for discussion section

## Evaluation Data

The `results/` folder contains:
- **Raw evaluation data**: `evaluation_data_*.csv` files with performance measurements for each system  
- **Generated LaTeX tables**: `RQ*_results.tex` files used in the paper
- **Cross-model analysis**: Introduction tables for each system
- **Discussion analysis**: Results from discussion analysis script
