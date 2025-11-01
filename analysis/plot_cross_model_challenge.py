#!/usr/bin/env python3
"""
Cross-Model Challenge Analysis Script

This script analyzes evaluation data from CSV files to find the best example
of cross-model prompt engineering challenge by systematically testing optimal
prompts across different LLMs.

Author: Meta Artemis Team
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
import logging

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

def find_optimal_prompts_per_project(df: pd.DataFrame) -> Dict:
    """
    Find optimal prompts for each LLM in each project.
    
    Args:
        df: DataFrame with evaluation data
        
    Returns:
        Dictionary with project -> LLM -> optimal prompt info
    """
    logger.info("Finding optimal prompts for each LLM in each project...")
    
    # Filter for single spec solutions
    single_spec_df = df[df['num_specs_in_solution'] == 1].copy()
    
    # Group by project and construct
    project_constructs = single_spec_df.groupby(['project_name', 'construct_id'])
    
    optimal_results = {}
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    
    for (project_name, construct_id), group in project_constructs:
        if project_name not in optimal_results:
            optimal_results[project_name] = {}
        
        if construct_id not in optimal_results[project_name]:
            optimal_results[project_name][construct_id] = {}
        
        # Find best solution for each LLM
        for llm in target_llms:
            llm_solutions = group[group['code_optimization_llm'] == llm]
            if llm_solutions.empty:
                continue
                
            # Sort all solutions by runtime to get best, second best, etc.
            sorted_solutions = []
            for _, solution in llm_solutions.iterrows():
                runtimes = parse_runtime_measurements(solution['runtime_measurements'])
                if runtimes:
                    min_runtime = np.min(runtimes)
                    sorted_solutions.append((min_runtime, solution))
            
            if sorted_solutions:
                # Sort by runtime (best first)
                sorted_solutions.sort(key=lambda x: x[0])
                
                # Store all solutions for this LLM (best, second best, etc.)
                optimal_results[project_name][construct_id][llm] = {
                    'solutions': sorted_solutions,  # List of (runtime, solution) tuples
                    'best_solution': sorted_solutions[0][1],
                    'best_prompt_version': sorted_solutions[0][1]['prompt_version'],
                    'best_mean_runtime': sorted_solutions[0][0],
                    'best_runtimes': parse_runtime_measurements(sorted_solutions[0][1]['runtime_measurements'])
                }
    
    return optimal_results

def find_cross_model_performance(df: pd.DataFrame, optimal_results: Dict) -> Dict:
    """
    Find cross-model performance using the best available prompts that have cross-model data.
    
    Args:
        df: DataFrame with evaluation data
        optimal_results: Dictionary with optimal prompts per project/construct/LLM
        
    Returns:
        Dictionary with cross-model performance data
    """
    logger.info("Finding cross-model performance with best available prompts...")
    
    cross_model_results = {}
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    
    for project_name, project_data in optimal_results.items():
        cross_model_results[project_name] = {}
        
        for construct_id, construct_data in project_data.items():
            cross_model_results[project_name][construct_id] = {}
            
            # For each LLM, find the best prompt that has cross-model data
            for source_llm in target_llms:
                if source_llm not in construct_data:
                    continue
                
                # Try each prompt for this LLM (best first, then second best, etc.)
                source_llm_data = construct_data[source_llm]
                best_available_prompt = None
                best_available_runtime = None
                
                for runtime, solution in source_llm_data['solutions']:
                    prompt_version = solution['prompt_version']
                    
                    # Check if this prompt has cross-model data for all target LLMs
                    has_complete_data = True
                    cross_model_data = {}
                    
                    for target_llm in target_llms:
                        if target_llm == source_llm:
                            # Use the solution's own performance
                            cross_model_data[target_llm] = {
                                'prompt_version': prompt_version,
                                'mean_runtime': runtime,
                                'runtimes': parse_runtime_measurements(solution['runtime_measurements']),
                                'is_optimal': True
                            }
                        else:
                            # Find cross-model performance
                            cross_model_solutions = df[
                                (df['project_name'] == project_name) &
                                (df['construct_id'] == construct_id) &
                                (df['code_optimization_llm'] == target_llm) &
                                (df['prompt_version'] == prompt_version) &
                                (df['num_specs_in_solution'] == 1)
                            ]
                            
                            if not cross_model_solutions.empty:
                                target_solution = cross_model_solutions.iloc[0]
                                target_runtimes = parse_runtime_measurements(target_solution['runtime_measurements'])
                                if target_runtimes:
                                    cross_model_data[target_llm] = {
                                        'prompt_version': prompt_version,
                                        'mean_runtime': np.min(target_runtimes),
                                        'runtimes': target_runtimes,
                                        'is_optimal': False
                                    }
                                else:
                                    has_complete_data = False
                                    break
                            else:
                                has_complete_data = False
                                break
                    
                    # If this prompt has complete cross-model data, use it
                    if has_complete_data:
                        best_available_prompt = prompt_version
                        best_available_runtime = runtime
                        cross_model_results[project_name][construct_id][source_llm] = cross_model_data
                        break
                
                # If no prompt has complete data, use the best available prompt with partial data
                if best_available_prompt is None:
                    # Find the prompt with the most cross-model data
                    best_coverage = 0
                    best_partial_data = None
                    best_partial_prompt = None
                    
                    for runtime, solution in source_llm_data['solutions']:
                        prompt_version = solution['prompt_version']
                        partial_data = {}
                        coverage = 0
                        
                        for target_llm in target_llms:
                            if target_llm == source_llm:
                                partial_data[target_llm] = {
                                    'prompt_version': prompt_version,
                                    'mean_runtime': runtime,
                                    'runtimes': parse_runtime_measurements(solution['runtime_measurements']),
                                    'is_optimal': True
                                }
                                coverage += 1
                            else:
                                cross_model_solutions = df[
                                    (df['project_name'] == project_name) &
                                    (df['construct_id'] == construct_id) &
                                    (df['code_optimization_llm'] == target_llm) &
                                    (df['prompt_version'] == prompt_version) &
                                    (df['num_specs_in_solution'] == 1)
                                ]
                                
                                if not cross_model_solutions.empty:
                                    target_solution = cross_model_solutions.iloc[0]
                                    target_runtimes = parse_runtime_measurements(target_solution['runtime_measurements'])
                                    if target_runtimes:
                                        partial_data[target_llm] = {
                                            'prompt_version': prompt_version,
                                            'mean_runtime': np.min(target_runtimes),
                                            'runtimes': target_runtimes,
                                            'is_optimal': False
                                        }
                                        coverage += 1
                        
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_partial_data = partial_data
                            best_partial_prompt = prompt_version
                    
                    if best_partial_data:
                        cross_model_results[project_name][construct_id][source_llm] = best_partial_data
    
    return cross_model_results

def analyze_cross_model_challenge(optimal_results: Dict, cross_model_results: Dict) -> Dict:
    """
    Analyze and find the best example of cross-model prompt engineering challenge.
    
    Args:
        optimal_results: Dictionary with optimal prompts
        cross_model_results: Dictionary with cross-model performance
        
    Returns:
        Dictionary with the best challenge example
    """
    logger.info("Analyzing cross-model challenge patterns...")
    
    best_example = None
    max_performance_gap = 0
    max_data_completeness = 0
    
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    
    for project_name, project_data in optimal_results.items():
        for construct_id, construct_data in project_data.items():
            if construct_id not in cross_model_results.get(project_name, {}):
                continue
                
            cross_model_data = cross_model_results[project_name][construct_id]
            
            # Check if we have all 3 models with optimal results
            available_models = [llm for llm in target_llms if llm in construct_data]
            if len(available_models) < 3:
                continue
            
            # Check if the cross-model matrix is fully filled (no N/A)
            fully_filled = True
            for source_llm in target_llms:
                if source_llm not in cross_model_data:
                    fully_filled = False
                    break
                for target_llm in target_llms:
                    if target_llm not in cross_model_data[source_llm]:
                        fully_filled = False
                        break
                if not fully_filled:
                    break
            if not fully_filled:
                continue
            
            # Calculate data completeness (should be 1.0 for fully filled)
            data_completeness = 1.0
            
            # Calculate total performance gap across all combinations
            total_gap = 0
            valid_combinations = 0
            for source_llm in target_llms:
                for target_llm in target_llms:
                    target_performance = cross_model_data[source_llm][target_llm]['mean_runtime']
                    target_optimal = construct_data[target_llm]['best_mean_runtime']
                    if target_optimal > 0:
                        gap = (target_performance - target_optimal) / target_optimal
                        total_gap += gap
                        valid_combinations += 1
            if valid_combinations > 0:
                avg_gap = total_gap / valid_combinations
                score = data_completeness * avg_gap
                if score > max_performance_gap:
                    max_performance_gap = score
                    max_data_completeness = data_completeness
                    best_example = {
                        'project_name': project_name,
                        'construct_id': construct_id,
                        'optimal_results': construct_data,
                        'cross_model_results': cross_model_data,
                        'performance_gap': avg_gap,
                        'data_completeness': data_completeness
                    }
    if best_example:
        logger.info(f"Best cross-model challenge example found:")
        logger.info(f"Project: {best_example['project_name']}")
        logger.info(f"Construct: {best_example['construct_id']}")
        logger.info(f"Average Performance Gap: {best_example['performance_gap']:.2%}")
        logger.info(f"Data Completeness: {best_example['data_completeness']:.1%}")
    return best_example

def print_detailed_analysis(optimal_results: Dict, cross_model_results: Dict):
    """
    Print detailed analysis of all projects and constructs.
    
    Args:
        optimal_results: Dictionary with optimal prompts
        cross_model_results: Dictionary with cross-model performance
    """
    logger.info("\n" + "="*80)
    logger.info("DETAILED CROSS-MODEL ANALYSIS")
    logger.info("="*80)
    
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    
    for project_name, project_data in optimal_results.items():
        logger.info(f"\nPROJECT: {project_name}")
        logger.info("-" * 50)
        
        for construct_id, construct_data in project_data.items():
            logger.info(f"\nConstruct: {construct_id}")
            
            # Print optimal performances
            logger.info("Optimal Performances:")
            for llm in target_llms:
                if llm in construct_data:
                    optimal = construct_data[llm]
                    logger.info(f"  {llm}: {optimal['best_mean_runtime']:.2f}s ({optimal['best_prompt_version']})")
            
            # Print cross-model performances
            if construct_id in cross_model_results.get(project_name, {}):
                cross_data = cross_model_results[project_name][construct_id]
                logger.info("\nCross-Model Performance:")
                
                for source_llm in target_llms:
                    if source_llm in cross_data:
                        # Find which prompt was actually used for this source LLM
                        used_prompt = None
                        for target_llm in target_llms:
                            if target_llm in cross_data[source_llm]:
                                used_prompt = cross_data[source_llm][target_llm]['prompt_version']
                                break
                        
                        if used_prompt:
                            logger.info(f"\n  Using {source_llm}'s prompt ({used_prompt}):")
                            for target_llm in target_llms:
                                if target_llm in cross_data[source_llm]:
                                    perf = cross_data[source_llm][target_llm]
                                    optimal_perf = construct_data[target_llm]['best_mean_runtime']
                                    gap = (perf['mean_runtime'] - optimal_perf) / optimal_perf * 100
                                    logger.info(f"    {target_llm}: {perf['mean_runtime']:.2f}s (gap: {gap:+.1f}%)")

def create_bar_chart_from_rankings(df: pd.DataFrame, output_file: str = "challenge_example.pdf"):
    """
    Create a bar chart showing each model's performance with their best ranked prompts.
    
    Args:
        df: DataFrame with evaluation data
        output_file: Output PDF file path
    """
    logger.info("Creating bar chart showing cross-model prompt performance using best ranked results...")
    
    # Font size configuration
    FONT_SIZE = 11
    
    # Set up the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    # Prepare data
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    llm_names = ['GPT-4o', 'Claude-3.7-Sonnet', 'Gemini-2.5-Pro']
    llm_names_short = ['GPT', 'Claude', 'Gemini']  # Short names for legend
    colors = ['#7C74FE', '#CCC9FF', '#007FFF']
    
    # Get the best prompt for each LLM and their cross-model performance
    best_prompts = {}
    cross_model_data = {}
    
    for llm in target_llms:
        # Get all prompt versions for this LLM
        llm_df = df[df['code_optimization_llm'] == llm]
        prompt_runtimes = []
        
        for prompt_version in llm_df['prompt_version'].unique():
            prompt_rows = llm_df[llm_df['prompt_version'] == prompt_version]
            all_runtimes = []
            for _, row in prompt_rows.iterrows():
                all_runtimes.extend(parse_runtime_measurements(row['runtime_measurements']))
            if all_runtimes:
                min_runtime = np.min(all_runtimes)
                prompt_runtimes.append((min_runtime, prompt_version))
        
        # Sort by best runtime
        prompt_runtimes.sort()
        
        # Find the first prompt that has complete cross-model data
        selected_prompt = None
        selected_runtime = None
        
        for runtime, prompt_version in prompt_runtimes:
            # Check if this prompt has data for all target LLMs
            has_complete_data = True
            temp_cross_data = {}
            
            for target_llm in target_llms:
                target_df = df[(df['code_optimization_llm'] == target_llm) & 
                              (df['prompt_version'] == prompt_version)]
                target_runtimes = []
                for _, row in target_df.iterrows():
                    target_runtimes.extend(parse_runtime_measurements(row['runtime_measurements']))
                
                if target_runtimes:
                    temp_cross_data[target_llm] = np.min(target_runtimes)
                else:
                    has_complete_data = False
                    break
            
            # If this prompt has complete data, use it
            if has_complete_data:
                selected_prompt = prompt_version
                selected_runtime = runtime
                cross_model_data[llm] = temp_cross_data
                break
        
        if selected_prompt:
            best_prompts[llm] = selected_prompt
            logger.info(f"Selected '{selected_prompt}' for {llm} (runtime: {selected_runtime:.1f}s) - has complete cross-model data")
        else:
            logger.warning(f"No prompt with complete cross-model data found for {llm}")
            # Fallback to best prompt even if incomplete
            if prompt_runtimes:
                best_prompts[llm] = prompt_runtimes[0][1]
                logger.info(f"Using best prompt '{prompt_runtimes[0][1]}' for {llm} despite incomplete data")
                
                # Get cross-model performance for this best prompt
                cross_model_data[llm] = {}
                for target_llm in target_llms:
                    target_df = df[(df['code_optimization_llm'] == target_llm) & 
                                  (df['prompt_version'] == best_prompts[llm])]
                    target_runtimes = []
                    for _, row in target_df.iterrows():
                        target_runtimes.extend(parse_runtime_measurements(row['runtime_measurements']))
                    
                    if target_runtimes:
                        cross_model_data[llm][target_llm] = np.min(target_runtimes)
                    else:
                        cross_model_data[llm][target_llm] = None
    
    # Create data matrix for plotting
    data_matrix = []
    has_data_matrix = []
    
    for target_llm in target_llms:
        row = []
        has_data_row = []
        for source_llm in target_llms:
            if source_llm in cross_model_data and target_llm in cross_model_data[source_llm]:
                runtime = cross_model_data[source_llm][target_llm]
                if runtime is not None:
                    row.append(runtime)
                    has_data_row.append(True)
                else:
                    row.append(0)
                    has_data_row.append(False)
            else:
                row.append(0)
                has_data_row.append(False)
        data_matrix.append(row)
        has_data_matrix.append(has_data_row)
    
    # Create bar positions
    x = np.arange(len(llm_names))
    width = 0.27  # Increased from 0.25 to make bars wider
    
    # Create bars for each prompt type
    bars = []
    prompt_names = []
    
    for i, source_llm in enumerate(target_llms):
        if source_llm in best_prompts:
            prompt_name = f"{llm_names_short[i]} Best Prompt"
            
            values = [data_matrix[j][i] for j in range(len(target_llms))]
            has_data = [has_data_matrix[j][i] for j in range(len(target_llms))]
            
            # Only create bars where we have data
            valid_values = []
            valid_positions = []
            for j, (value, has_data_point) in enumerate(zip(values, has_data)):
                if has_data_point:
                    valid_values.append(value)
                    valid_positions.append(x[j] + i * width)
            
            if valid_values:
                bar = ax.bar(valid_positions, valid_values, width, label=prompt_name, color=colors[i], alpha=0.8)
                bars.append(bar)
    
    # Customize the plot
    ax.set_xlabel('Code Optimization LLM', fontsize=FONT_SIZE, fontweight='normal')
    ax.set_ylabel('Runtime (seconds)', fontsize=FONT_SIZE, fontweight='normal')
    ax.set_xticks(x + width)
    ax.set_xticklabels(llm_names, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE-2, loc='upper center', bbox_to_anchor=(0.48, 1.15), ncol=3)
    
    # Add value labels on bars
    def add_value_labels(bars_list):
        for bars_group in bars_list:
            for bar in bars_group:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=FONT_SIZE-2)
    
    add_value_labels(bars)
    
    # Remove the data source note
    
    # Customize grid
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    max_runtime = max([max(row) for row in data_matrix if any(row)])
    min_runtime = min([min(row) for row in data_matrix if any(row)])
    ax.set_ylim(bottom=min_runtime * 0.95, top=max_runtime * 1.02)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2E2E2E')
    ax.spines['bottom'].set_color('#2E2E2E')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    logger.info(f"Saving plot to {output_file}")
    plt.savefig(output_file, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    logger.info("Bar chart created and saved successfully!")
    
    # Print the data used for the chart
    logger.info("\nData used for the chart:")
    logger.info("=" * 60)
    for source_llm in target_llms:
        if source_llm in best_prompts:
            logger.info(f"\n{llm_names[target_llms.index(source_llm)]} using '{best_prompts[source_llm]}':")
            for target_llm in target_llms:
                if source_llm in cross_model_data and target_llm in cross_model_data[source_llm]:
                    runtime = cross_model_data[source_llm][target_llm]
                    if runtime is not None:
                        logger.info(f"  -> {llm_names[target_llms.index(target_llm)]}: {runtime:.1f}s")
                    else:
                        logger.info(f"  -> {llm_names[target_llms.index(target_llm)]}: No data")
                else:
                    logger.info(f"  -> {llm_names[target_llms.index(target_llm)]}: No data")

def print_top_prompt_runtimes(df, top_n=20):
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    print("\n=== Top Prompt Templates by LLM (sorted by best runtime) ===")
    for llm in target_llms:
        print(f"\nLLM: {llm}")
        # Get all prompt versions for this LLM
        llm_df = df[df['code_optimization_llm'] == llm]
        prompt_runtimes = []
        for prompt_version in llm_df['prompt_version'].unique():
            prompt_rows = llm_df[llm_df['prompt_version'] == prompt_version]
            all_runtimes = []
            for _, row in prompt_rows.iterrows():
                all_runtimes.extend(parse_runtime_measurements(row['runtime_measurements']))
            if all_runtimes:
                min_runtime = np.min(all_runtimes)
                prompt_runtimes.append((min_runtime, prompt_version))
        # Sort by best runtime
        prompt_runtimes.sort()
        print(f"{'Prompt':<30} {'This LLM':>10} {'Other1':>10} {'Other2':>10}")
        print("-" * 60)
        for min_runtime, prompt_version in prompt_runtimes[:top_n]:
            row = [f"{prompt_version:<30} {min_runtime:>8.1f}s"]
            # For other LLMs, get min runtime for this prompt (if available)
            for other_llm in [x for x in target_llms if x != llm]:
                other_df = df[(df['code_optimization_llm'] == other_llm) & (df['prompt_version'] == prompt_version)]
                other_runtimes = []
                for _, row2 in other_df.iterrows():
                    other_runtimes.extend(parse_runtime_measurements(row2['runtime_measurements']))
                if other_runtimes:
                    row.append(f"{np.min(other_runtimes):>8.1f}s")
                else:
                    row.append(f"{'--':>8}")
            print(" ".join(row))

def main():
    """
    Main function to run the cross-model challenge analysis.
    """
    logger.info("Starting Systematic Cross-Model Challenge Analysis")
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return
    
    # ============================================================================
    # EASY CONTROL: Change this variable to select which system to analyze
    # ============================================================================
    # Options: "llama.cpp", "rpcs3", "faster-whisper", "BitmapPlusPlus", "Langflow", "auto"
    # Set to "auto" to automatically find the best example across all systems
    SELECTED_SYSTEM = "llama.cpp"  # <-- CHANGE THIS LINE TO SELECT SYSTEM
    # ============================================================================
    
    # Define system files with their patterns
    system_patterns = {
        "llama.cpp": "evaluation_data_llama.cpp_*.csv",
        "rpcs3": "evaluation_data_rpcs3_*.csv", 
        "faster-whisper": "evaluation_data_faster-whisper_*.csv",
        "BitmapPlusPlus": "evaluation_data_BitmapPlusPlus_*.csv",
        "Langflow": "evaluation_data_Langflow_*.csv"
    }
    
    latest_csv = None
    
    if SELECTED_SYSTEM == "auto":
        # Try all systems and find the best example
        logger.info("Auto mode: Will analyze all available systems to find the best example")
        all_system_files = []
        for system_name, pattern in system_patterns.items():
            system_files = list(results_dir.glob(pattern))
            if system_files:
                latest_system_file = max(system_files, key=lambda x: x.stat().st_mtime)
                all_system_files.append((system_name, latest_system_file))
                logger.info(f"Found {system_name}: {latest_system_file}")
        
        if not all_system_files:
            logger.error("No system files found")
            return
        
        # Analyze each system and find the best overall example
        best_overall_example = None
        best_overall_score = 0
        
        for system_name, system_file in all_system_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"ANALYZING SYSTEM: {system_name}")
            logger.info(f"{'='*60}")
            
            # Load data for this system
            df = load_evaluation_data(str(system_file))
            if df.empty:
                logger.warning(f"Failed to load data for {system_name}, skipping...")
                continue
            
            # Find optimal prompts for this system
            optimal_results = find_optimal_prompts_per_project(df)
            cross_model_results = find_cross_model_performance(df, optimal_results)
            
            # Find best example for this system
            best_example = analyze_cross_model_challenge(optimal_results, cross_model_results)
            
            if best_example:
                score = best_example['data_completeness'] * best_example['performance_gap']
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_example = {
                        'system_name': system_name,
                        'system_file': system_file,
                        'example': best_example
                    }
        
        if best_overall_example:
            logger.info(f"\n{'='*60}")
            logger.info(f"BEST OVERALL EXAMPLE FOUND IN: {best_overall_example['system_name']}")
            logger.info(f"{'='*60}")
            latest_csv = best_overall_example['system_file']
            best_example = best_overall_example['example']
            
            # Load the data for the best system to create the chart
            df = load_evaluation_data(str(latest_csv))
            if df.empty:
                logger.error("Failed to load evaluation data for best system")
                return
        else:
            logger.error("No suitable examples found across all systems")
            return
            
    else:
        # Use specific system
        if SELECTED_SYSTEM not in system_patterns:
            logger.error(f"Unknown system: {SELECTED_SYSTEM}")
            logger.info(f"Available systems: {list(system_patterns.keys())}")
            return
        
        pattern = system_patterns[SELECTED_SYSTEM]
        system_files = list(results_dir.glob(pattern))
        
        if not system_files:
            logger.error(f"No files found for system: {SELECTED_SYSTEM}")
            return
        
        latest_csv = max(system_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using {SELECTED_SYSTEM} CSV file: {latest_csv}")
        
        # Load data
        df = load_evaluation_data(str(latest_csv))
        if df.empty:
            logger.error("Failed to load evaluation data")
            return
        
        # Find optimal prompts for this system
        optimal_results = find_optimal_prompts_per_project(df)
        cross_model_results = find_cross_model_performance(df, optimal_results)
        
        # Find best example for this system
        best_example = analyze_cross_model_challenge(optimal_results, cross_model_results)
        
        if not best_example:
            logger.error(f"No suitable cross-model challenge example found for {SELECTED_SYSTEM}")
            return
    
    # For auto mode, we already have the best example and df loaded
    # For specific system mode, we need to print detailed analysis
    if SELECTED_SYSTEM != "auto":
        # Print detailed analysis
        print_detailed_analysis(optimal_results, cross_model_results)
    
    # Create bar chart using best ranked results
    create_bar_chart_from_rankings(df)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BEST CROSS-MODEL CHALLENGE EXAMPLE")
    logger.info("="*60)
    if SELECTED_SYSTEM == "auto":
        logger.info(f"System: {best_overall_example['system_name']}")
    logger.info(f"Project: {best_example['project_name']}")
    logger.info(f"Construct: {best_example['construct_id']}")
    logger.info(f"Average Performance Gap: {best_example['performance_gap']:.2%}")
    logger.info(f"Data Completeness: {best_example['data_completeness']:.1%}")
    
    logger.info("\nDetailed Performance Matrix:")
    target_llms = ['gpt-4-o', 'claude', 'gemini']
    llm_names = ['GPT-4o', 'Claude-3.7-Sonnet', 'Gemini-2.5-Pro']
    
    # Print header
    logger.info(f"{'Target Model':<20} {'GPT-4o Prompt':<15} {'Claude Prompt':<15} {'Gemini Prompt':<15}")
    logger.info("-" * 70)
    
    for i, target_llm in enumerate(target_llms):
        row = []
        for source_llm in target_llms:
            if source_llm in best_example['cross_model_results'] and target_llm in best_example['cross_model_results'][source_llm]:
                runtime = best_example['cross_model_results'][source_llm][target_llm]['mean_runtime']
                row.append(f"{runtime:.1f}s")
            else:
                row.append("N/A")
        
        logger.info(f"{llm_names[i]:<20} {row[0]:<15} {row[1]:<15} {row[2]:<15}")
    
    logger.info("="*60)

    print_top_prompt_runtimes(df, top_n=20)

if __name__ == "__main__":
    main() 