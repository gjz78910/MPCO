#!/usr/bin/env python3
"""
Intro LaTeX Table Generator

This script analyzes evaluation data from CSV files to generate LaTeX tables showing
how model-specific optimal prompts perform across different models.

The script:
1. Loads evaluation data from CSV files
2. Finds the best runtime for each LLM (GPT, Claude, Gemini) in a system
3. Identifies which prompt template leads to the best runtime for each LLM
4. Tests how each LLM performs with other LLMs' optimal prompts
5. Generates LaTeX tables showing cross-model performance for all systems

Author: Meta Artemis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_evaluation_data(csv_file: str) -> pd.DataFrame:
    """
    Load evaluation data from CSV file.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        DataFrame with evaluation data
    """
    logger.info(f"Loading evaluation data from {csv_file}")
    df = pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Successfully loaded {len(df)} rows from {csv_file}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return df

def parse_runtime_measurements(runtime_str: str) -> List[float]:
    """
    Parse runtime measurements string into list of floats.
    
    Args:
        runtime_str: Comma-separated string of runtime measurements
        
    Returns:
        List of float values
    """
    if pd.isna(runtime_str) or runtime_str == "":
        return []
    
    try:
        # Remove quotes and split by comma
        clean_str = runtime_str.strip('"')
        measurements = [float(x.strip()) for x in clean_str.split(',') if x.strip()]
        return measurements
    except Exception as e:
        logger.warning(f"Error parsing runtime measurements '{runtime_str}': {e}")
        return []

def get_llm_data(df: pd.DataFrame, llm_name: str) -> pd.DataFrame:
    """
    Get data for a specific LLM.
    
    Args:
        df: DataFrame with evaluation data
        llm_name: Name of the LLM (gpt-4-o, claude, gemini)
        
    Returns:
        DataFrame filtered for the specific LLM
    """
    # Map LLM names to their patterns in the data
    llm_patterns = {
        'gpt-4-o': 'gpt-4-o',
        'claude': 'claude',
        'gemini': 'gemini'
    }
    
    pattern = llm_patterns.get(llm_name.lower(), llm_name.lower())
    llm_data = df[df['code_optimization_llm'].str.contains(pattern, case=False, na=False)]
    
    return llm_data

def find_best_prompt_for_llm(df: pd.DataFrame, llm_name: str) -> Tuple[str, float]:
    """
    Find the best prompt template for a specific LLM.
    
    Args:
        df: DataFrame with evaluation data
        llm_name: Name of the LLM
        
    Returns:
        Tuple of (best_prompt_template, best_runtime)
    """
    llm_data = get_llm_data(df, llm_name)
    
    if llm_data.empty:
        logger.warning(f"No data found for {llm_name}")
        return None, float('inf')
    
    best_runtime = float('inf')
    best_prompt = None
    
    # Group by prompt_version to find the best one
    for prompt_version, group in llm_data.groupby('prompt_version'):
        all_runtimes = []
        
        for _, row in group.iterrows():
            runtimes = parse_runtime_measurements(row['runtime_measurements'])
            if runtimes:
                all_runtimes.extend(runtimes)
        
        if all_runtimes:
            avg_runtime = np.mean(all_runtimes)
            if avg_runtime < best_runtime:
                best_runtime = avg_runtime
                best_prompt = prompt_version
    
    return best_prompt, best_runtime

def get_runtime_for_prompt_llm_combination(df: pd.DataFrame, prompt_template: str, llm_name: str) -> float:
    """
    Get runtime for a specific prompt template and LLM combination.
    
    Args:
        df: DataFrame with evaluation data
        prompt_template: Prompt template to test
        llm_name: Name of the LLM
        
    Returns:
        Average runtime for the combination
    """
    llm_data = get_llm_data(df, llm_name)
    prompt_data = llm_data[llm_data['prompt_version'] == prompt_template]
    
    if prompt_data.empty:
        logger.warning(f"No data found for {prompt_template} with {llm_name}")
        return float('inf')
    
    all_runtimes = []
    for _, row in prompt_data.iterrows():
        runtimes = parse_runtime_measurements(row['runtime_measurements'])
        if runtimes:
            all_runtimes.extend(runtimes)
    
    if all_runtimes:
        return np.mean(all_runtimes)
    else:
        return float('inf')

def find_best_prompts_for_cross_model_testing(df: pd.DataFrame) -> Dict[str, str]:
    """
    Find the best prompts for each LLM that have data across multiple LLMs.
    
    Args:
        df: DataFrame with evaluation data
        
    Returns:
        Dictionary mapping LLM to their best available prompt for cross-model testing
    """
    llms = ['gpt-4-o', 'claude', 'gemini']
    best_prompts = {}
    
    # First, find the absolute best prompt for each LLM
    absolute_best_prompts = {}
    for llm in llms:
        best_prompt, best_runtime = find_best_prompt_for_llm(df, llm)
        if best_prompt:
            absolute_best_prompts[llm] = best_prompt
            logger.info(f"Absolute best prompt for {llm}: {best_prompt} (runtime: {best_runtime:.4f})")
    
    # Find all available prompts for each LLM
    available_prompts = {}
    for llm in llms:
        llm_data = get_llm_data(df, llm)
        if not llm_data.empty:
            available_prompts[llm] = set(llm_data['prompt_version'].unique())
    
    # Find prompts that are available across multiple LLMs
    cross_model_prompts = set()
    all_prompts = set()
    for prompts in available_prompts.values():
        all_prompts.update(prompts)
    
    for prompt in all_prompts:
        llm_count = sum(1 for llm in llms if llm in available_prompts and prompt in available_prompts[llm])
        if llm_count >= 2:
            cross_model_prompts.add(prompt)
    
    # For each LLM, find the best prompt among cross-model compatible prompts
    # If the absolute best is cross-model compatible, use it
    # Otherwise, find the best cross-model compatible prompt
    for llm in llms:
        if llm not in available_prompts:
            continue
            
        best_runtime = float('inf')
        best_prompt = None
        
        # First, try to use the absolute best prompt if it's cross-model compatible
        if llm in absolute_best_prompts:
            absolute_best = absolute_best_prompts[llm]
            if absolute_best in cross_model_prompts:
                runtime = get_runtime_for_prompt_llm_combination(df, absolute_best, llm)
                if runtime != float('inf'):
                    best_prompts[llm] = absolute_best
                    logger.info(f"Using absolute best cross-model prompt for {llm}: {absolute_best} (runtime: {runtime:.4f})")
                    continue
        
        # Otherwise, find the best cross-model compatible prompt
        for prompt in cross_model_prompts:
            if prompt in available_prompts[llm]:
                runtime = get_runtime_for_prompt_llm_combination(df, prompt, llm)
                if runtime < best_runtime:
                    best_runtime = runtime
                    best_prompt = prompt
        
        if best_prompt:
            best_prompts[llm] = best_prompt
            logger.info(f"Best cross-model prompt for {llm}: {best_prompt} (runtime: {best_runtime:.4f})")
    
    return best_prompts

def analyze_cross_model_performance(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze cross-model performance for a system.
    
    Args:
        df: DataFrame with evaluation data
        
    Returns:
        Dictionary with cross-model performance data
    """
    logger.info("Analyzing cross-model performance...")
    
    llms = ['gpt-4-o', 'claude', 'gemini']
    results = {}
    
    # Find best prompts that have cross-model data
    best_prompts = find_best_prompts_for_cross_model_testing(df)
    
    # Test each LLM with each prompt
    for llm in llms:
        results[llm] = {}
        for prompt_llm in llms:
            if prompt_llm in best_prompts:
                prompt_template = best_prompts[prompt_llm]
                runtime = get_runtime_for_prompt_llm_combination(df, prompt_template, llm)
                if runtime != float('inf'):
                    results[llm][prompt_llm] = runtime
                    logger.info(f"{llm} with {prompt_llm}'s best prompt: {runtime:.4f}")
                else:
                    logger.warning(f"No data for {llm} with {prompt_llm}'s prompt ({prompt_template})")
    
    return results

def calculate_cross_model_variance(results: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate variance in cross-model performance to measure how dramatic the differences are.
    
    Args:
        results: Cross-model performance results
        
    Returns:
        Variance value (higher = more dramatic differences)
    """
    all_runtimes = []
    for llm_results in results.values():
        for runtime in llm_results.values():
            if runtime != float('inf'):
                all_runtimes.append(runtime)
    
    if len(all_runtimes) >= 2:
        return np.var(all_runtimes)
    return 0

def generate_intro_latex_table(results: Dict[str, Dict[str, float]], system_name: str, output_file: str):
    """
    Generate LaTeX table for intro cross-model performance.
    
    Args:
        results: Dictionary with cross-model performance results
        system_name: Name of the system
        output_file: Output file path
    """
    logger.info(f"Generating LaTeX table for {system_name}...")
    
    # Define LLM display names
    llm_display_names = {
        'gpt-4-o': 'GPT-4o',
        'claude': 'Claude-3.7-Sonnet',
        'gemini': 'Gemini-2.5-Pro'
    }
    
    # Calculate variance for caption
    variance = calculate_cross_model_variance(results)
    
    latex_content = f"""\\begin{{table}}[t!]
    \\footnotesize
    \\centering
    \\caption{{Cross-model performance analysis for {system_name} (variance: {variance:.2f}). Each cell shows runtime (seconds) when an LLM uses another LLM's optimal prompt. Lower values indicate better performance.}}
    \\label{{tab:intro_cross_model_{system_name.lower().replace('-', '_').replace('.', '_')}}}
    \\setlength{{\\tabcolsep}}{{3mm}}
    \\begin{{adjustbox}}{{width=0.9\\columnwidth,center}}
    \\begin{{tabular}}{{lccc}}
    \\toprule
    \\textbf{{LLM}} & \\textbf{{GPT's}} & \\textbf{{Claude's}} & \\textbf{{Gemini's}} \\\\
    & \\textbf{{prompt}} & \\textbf{{prompt}} & \\textbf{{prompt}} \\\\
    \\hline
"""
    
    # Generate table rows
    for llm in ['gpt-4-o', 'claude', 'gemini']:
        display_name = llm_display_names[llm]
        row = f"    {display_name}"
        
        for prompt_llm in ['gpt-4-o', 'claude', 'gemini']:
            if llm in results and prompt_llm in results[llm]:
                runtime = results[llm][prompt_llm]
                if runtime == float('inf'):
                    row += " & --"
                else:
                    row += f" & {runtime:.1f}"
            else:
                row += " & --"
        
        latex_content += f"{row} \\\\\n"
    
    latex_content += """    \\bottomrule
    \\end{tabular}
    \\end{adjustbox}
    \\end{table}"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    logger.info(f"LaTeX table saved to {output_file}")

def main():
    """
    Main function to generate intro LaTeX tables for all systems.
    """
    logger.info("Starting intro LaTeX table generation for all systems")
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return
    
    # Process each system
    system_results = {}
    
    for csv_file in csv_files:
        df = load_evaluation_data(str(csv_file))
        if df.empty:
            continue
        
        system_name = df['project_name'].iloc[0]
        logger.info(f"\nProcessing system: {system_name}")
        
        # Analyze cross-model performance
        results = analyze_cross_model_performance(df)
        
        if results:
            # Calculate variance to rank systems by dramatic differences
            variance = calculate_cross_model_variance(results)
            system_results[system_name] = {
                'results': results,
                'variance': variance,
                'csv_file': csv_file
            }
            
            # Generate LaTeX table for this system
            output_file = f'results/intro_cross_model_table_{system_name.lower().replace("-", "_").replace(".", "_")}.tex'
            generate_intro_latex_table(results, system_name, output_file)
            
            logger.info(f"System {system_name} variance: {variance:.2f}")
    
    # Print summary of all systems
    if system_results:
        logger.info("\n" + "="*80)
        logger.info("SYSTEMS RANKED BY CROSS-MODEL VARIANCE (most dramatic differences first)")
        logger.info("="*80)
        
        # Sort systems by variance (descending)
        sorted_systems = sorted(system_results.items(), key=lambda x: x[1]['variance'], reverse=True)
        
        for i, (system_name, data) in enumerate(sorted_systems, 1):
            variance = data['variance']
            logger.info(f"{i}. {system_name}: variance = {variance:.2f}")
            
            # Print detailed results for top 3 systems
            if i <= 3:
                logger.info(f"   Results:")
                for llm, prompt_results in data['results'].items():
                    for prompt_llm, runtime in prompt_results.items():
                        if runtime != float('inf'):
                            logger.info(f"     {llm} with {prompt_llm}'s prompt: {runtime:.1f}s")
                        else:
                            logger.info(f"     {llm} with {prompt_llm}'s prompt: N/A")
                logger.info("")
        
        logger.info("LaTeX tables generated for all systems. Check the results/ directory.")
    else:
        logger.error("No valid results found for any system")

if __name__ == "__main__":
    main() 