#!/usr/bin/env python3
"""
RQs LaTeX Table Generator

This script analyzes evaluation data from CSV files to generate LaTeX tables for RQ1, RQ2, and RQ3
showing the effectiveness of MPCO compared to different approaches.

The script:
1. Loads evaluation data from CSV files
2. Calculates %PI (Percentage Performance Improvement) for each approach
3. Performs statistical tests (Mann-Whitney U test and Cohen's d effect size)
4. Ranks approaches based on statistical significance
5. Generates LaTeX tables for each RQ

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

def calculate_performance_improvement(original_runtimes: List[float], optimized_runtimes: List[float]) -> float:
    """
    Calculate percentage performance improvement.
    
    Args:
        original_runtimes: List of original runtime measurements
        optimized_runtimes: List of optimized runtime measurements
        
    Returns:
        Percentage improvement (positive = improvement, negative = degradation)
    """
    if not original_runtimes or not optimized_runtimes:
        return 0.0
    
    original_mean = np.mean(original_runtimes)
    optimized_mean = np.mean(optimized_runtimes)
    
    if original_mean == 0:
        return 0.0
    
    # %PI = (T_orig - T_opt) / T_orig * 100%
    improvement = (original_mean - optimized_mean) / original_mean * 100
    return improvement

def get_original_runtimes(df: pd.DataFrame, project_name: str) -> List[float]:
    """
    Get original runtime measurements for a project.
    
    Args:
        df: DataFrame with evaluation data
        project_name: Name of the project
        
    Returns:
        List of original runtime measurements
    """
    # Look for original data with different possible patterns
    original_data = df[
        (df['project_name'] == project_name) & 
        (df['prompt_version'] == 'original') &
        ((df['construct_id'] == 'All') | (df['construct_id'] == 'Original'))
    ]
    
    if original_data.empty:
        # Try alternative patterns
        original_data = df[
            (df['project_name'] == project_name) & 
            (df['prompt_version'] == 'original')
        ]
    
    if original_data.empty:
        logger.warning(f"No original data found for {project_name}")
        return []
    
    runtime_str = original_data.iloc[0]['runtime_measurements']
    return parse_runtime_measurements(runtime_str)

def calculate_pi_for_approach(df: pd.DataFrame, project_name: str, prompt_version: str, optimization_llm: str = None) -> List[float]:
    """
    Calculate PI values for all constructs of a specific approach in a project.
    
    Args:
        df: DataFrame with evaluation data
        project_name: Name of the project
        prompt_version: Prompt version to analyze
        optimization_llm: Optional optimization LLM filter (for RQ3)
        
    Returns:
        List of PI values for all constructs
    """
    original_runtimes = get_original_runtimes(df, project_name)
    if not original_runtimes:
        return []
    
    pi_values = []
    
    # Get all constructs for this project and prompt version
    approach_data = df[
        (df['project_name'] == project_name) & 
        (df['prompt_version'] == prompt_version) &
        (df['construct_id'] != 'All')  # Exclude the "All" construct
    ]
    
    # Add optimization LLM filter if specified (for RQ3)
    if optimization_llm:
        approach_data = approach_data[approach_data['code_optimization_llm'] == optimization_llm]
    
    # Group by construct_id to get all measurements for each construct
    for construct_id, group in approach_data.groupby('construct_id'):
        if group.empty:
            continue
            
        # Get all runtime measurements for this construct
        all_runtimes = []
        for _, row in group.iterrows():
            runtimes = parse_runtime_measurements(row['runtime_measurements'])
            if runtimes:
                all_runtimes.extend(runtimes)
        
        if all_runtimes:
            pi = calculate_performance_improvement(original_runtimes, all_runtimes)
            pi_values.append(pi)
    
    return pi_values

def perform_statistical_tests(approach_pi_values: Dict[str, List[float]]) -> Dict[str, Dict]:
    """
    Perform statistical tests between approaches.
    
    Args:
        approach_pi_values: Dictionary with approach names and their PI values
        
    Returns:
        Dictionary with statistical test results
    """
    logger.info("Performing statistical tests...")
    
    approaches = list(approach_pi_values.keys())
    results = {}
    
    for i, approach1 in enumerate(approaches):
        results[approach1] = {}
        values1 = approach_pi_values[approach1]
        
        if not values1:
            continue
            
        for j, approach2 in enumerate(approaches):
            if i == j:
                continue
                
            values2 = approach_pi_values[approach2]
            if not values2:
                continue
            
            # Mann-Whitney U test
            try:
                statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
            except:
                p_value = 1.0
            
            # Cohen's d effect size
            try:
                pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                    (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                   (len(values1) + len(values2) - 2))
                if pooled_std == 0:
                    cohens_d = 0
                else:
                    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
            except:
                cohens_d = 0
            
            results[approach1][approach2] = {
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value <= 0.05 or abs(cohens_d) >= 0.2
            }
    
    return results

def rank_approaches(approach_pi_values: Dict[str, List[float]], 
                   statistical_results: Dict[str, Dict]) -> Dict[str, int]:
    """
    Rank approaches based on statistical significance.
    
    Args:
        approach_pi_values: Dictionary with approach names and their PI values
        statistical_results: Dictionary with statistical test results
        
    Returns:
        Dictionary with approach names and their ranks
    """
    logger.info("Ranking approaches based on statistical significance...")
    
    approaches = list(approach_pi_values.keys())
    ranks = {}
    
    # Calculate mean PI for each approach
    mean_pi = {}
    for approach in approaches:
        values = approach_pi_values[approach]
        if values:
            mean_pi[approach] = np.mean(values)
        else:
            mean_pi[approach] = 0
    
    # Sort by mean PI (descending)
    sorted_approaches = sorted(mean_pi.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize ranks
    current_rank = 1
    for i, (approach1, mean1) in enumerate(sorted_approaches):
        if approach1 not in ranks:
            ranks[approach1] = current_rank
        
        # Check if this approach is significantly different from the next one
        for j in range(i + 1, len(sorted_approaches)):
            approach2, mean2 = sorted_approaches[j]
            
            if approach2 not in ranks:
                # Check if approaches are significantly different
                if (approach1 in statistical_results and 
                    approach2 in statistical_results[approach1] and
                    statistical_results[approach1][approach2]['significant']):
                    # They are significantly different, so different ranks
                    ranks[approach2] = current_rank + 1
                    current_rank += 1
                else:
                    # They are not significantly different, so same rank
                    ranks[approach2] = current_rank
        
        current_rank += 1
    
    return ranks

def get_rq_config(rq: str, use_median: bool = False) -> Dict:
    """
    Get configuration for a specific RQ.
    
    Args:
        rq: RQ number ('rq1', 'rq2', or 'rq3')
        use_median: Whether to use median/IQR instead of mean/SD
        
    Returns:
        Dictionary with RQ configuration
    """
    if use_median:
        stat_name = "median and interquartile range"
        stat_abbrev = "Median (IQR)"
        best_stat = "median"
    else:
        stat_name = "mean and standard deviation"
        stat_abbrev = "Mean (SD)"
        best_stat = "mean"
    
    configs = {
        'rq1': {
            'approaches': ['mpco', 'chain_of_thought', 'few_shot', 'contextual_prompting'],
            'approach_names': ['\\model', 'CoT', 'Few-shot', 'Contextual'],
            'caption': f'The {stat_name} of \\%PI, denoted as {stat_abbrev}, for \\model~and the basic prompting approach across three LLMs and five projects. For each case, \\setlength{{\\fboxsep}}{{1.5pt}}\\colorbox{{applegreen!30}}{{green cells}} mean \\model~has the best {best_stat} \\%PI; or \\setlength{{\\fboxsep}}{{1.5pt}}\\colorbox{{red!20}}{{red cells}} otherwise. The one(s) with the best rank ($r$) from the statistical tests is in bold.',
            'label': 'tab:rq1_results',
            'output_file': 'results/RQ1_results.tex'
        },
        'rq2': {
            'approaches': ['mpco', 'no_project_context', 'no_task_context', 'no_llm_context'],
            'approach_names': ['\\model', '\\model$_{\\texttt{NP}}$', '\\model$_{\\texttt{NT}}$', '\\model$_{\\texttt{NL}}$'],
            'caption': f'The {stat_name} of \\%PI, denoted as {stat_abbrev}, for \\model~and its ablated versions across three LLMs and five projects. For each case, \\setlength{{\\fboxsep}}{{1.5pt}}\\colorbox{{applegreen!30}}{{green cells}} mean \\model~has the best {best_stat} \\%PI; or \\setlength{{\\fboxsep}}{{1.5pt}}\\colorbox{{red!20}}{{red cells}} otherwise. The one(s) with the best rank ($r$) from the statistical tests is in bold.',
            'label': 'tab:rq2_results',
            'output_file': 'results/RQ2_results.tex'
        },
        'rq3': {
            'approaches': ['mpco', 'mpco2', 'mpco3'],
            'approach_names': ['\\model$_{\\texttt{4o}}$', '\\model$_{\\texttt{37}}$', '\\model$_{\\texttt{25}}$'],
            'optimization_llms': ['claude', 'claude', 'claude'],
            'caption': f'The {stat_name} of \\%PI, denoted as {stat_abbrev}, for different meta-prompting LLMs across five projects. For each case, \\setlength{{\\fboxsep}}{{1.5pt}}\\colorbox{{applegreen!30}}{{green cells}} mean the approach has the best {best_stat} \\%PI. The one(s) with the best rank ($r$) from the statistical tests is in bold.',
            'label': 'tab:rq3_results',
            'output_file': 'results/RQ3_results.tex'
        }
    }
    
    return configs.get(rq.lower(), configs['rq1'])

def generate_latex_table(results: Dict[str, Dict], rq: str, use_median: bool = False):
    """
    Generate LaTeX table for RQ results.
    
    Args:
        results: Dictionary with results for each system
        rq: RQ number ('rq1', 'rq2', or 'rq3')
        use_median: Whether to use median/IQR instead of mean/SD
    """
    logger.info(f"Generating LaTeX table for {rq.upper()}...")
    
    # Get RQ configuration
    config = get_rq_config(rq, use_median)
    approaches = config['approaches']
    approach_names = config['approach_names']
    caption = config['caption']
    label = config['label']
    output_file = config['output_file']
    
    # Define the systems and approaches based on the table template
    # Map CSV project names to display names
    system_mapping = {
        'BitmapPlusPlus': 'BitmapPlusPlus',
        'llama.cpp': 'Llama.cpp',
        'rpcs3': 'RPCS3',
        'faster-whisper': 'Faster‑Whisper', 
        'Langflow': 'Langflow'
    }
    systems = list(system_mapping.keys())  # Use actual project names from CSV
    
    latex_content = f"""\\begin{{table}}[t!]
    \\footnotesize
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\setlength{{\\tabcolsep}}{{0.6mm}}
    \\begin{{adjustbox}}{{width=\\columnwidth,center}}
    \\begin{{tabular}}{{{'l' * (len(approaches) * 2 + 1)}}}
    \\toprule
    \\multirow{{2}}{{*}}{{\\textbf{{System}}}}"""
    
    # Add column headers
    for approach_name in approach_names:
        latex_content += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{approach_name}}}}}"
    
    latex_content += " \\\\ \\cline{2-" + str(len(approaches) * 2 + 1) + "}\n"
    
    # Add subheaders
    if use_median:
        stat_header = "Median (IQR)"
    else:
        stat_header = "Mean (SD)"
    
    for _ in approaches:
        latex_content += f" & \\textbf{{$r$}} & \\textbf{{{stat_header}}}"
    
    latex_content += " \\\\ \\hline\n"
    
    # Generate table rows
    for system in systems:
        if system not in results:
            continue
            
        system_data = results[system]
        # Use the display name from the mapping
        display_name = system_mapping.get(system, system)
        row = f"\\textsc{{{display_name}}}"
        
        for approach, approach_name in zip(approaches, approach_names):
            if approach in system_data:
                data = system_data[approach]
                rank = data['rank']
                
                if use_median:
                    median_pi = data['median_pi']
                    iqr_pi = data['iqr_pi']
                    stat_value = median_pi
                    stat_dispersion = iqr_pi
                    # Determine which approach has the best median %PI for this system
                    best_stat_approach = max(system_data.keys(), 
                                           key=lambda x: system_data[x]['median_pi'])
                else:
                    mean_pi = data['mean_pi']
                    std_pi = data['std_pi']
                    stat_value = mean_pi
                    stat_dispersion = std_pi
                    # Determine which approach has the best mean %PI for this system
                    best_stat_approach = max(system_data.keys(), 
                                           key=lambda x: system_data[x]['mean_pi'])
                
                # Determine if this approach has rank 1 (best rank) for this system
                is_best_rank = rank == 1
                
                # Format the cell
                rank_part = f"\\textbf{{{rank}}}" if is_best_rank else f"{rank}"
                stat_part = f"\\textbf{{{stat_value:.2f}}} ({stat_dispersion:.2f})" if is_best_rank else f"{stat_value:.2f} ({stat_dispersion:.2f})"
                
                # Color logic: only color the approach with the best statistic %PI
                if approach == best_stat_approach:
                    if rq.lower() == 'rq3':
                        # For RQ3, all approaches are MPCO variants, so color all best approaches green
                        cell = f"\\cellcolor{{applegreen!30}}{rank_part} & \\cellcolor{{applegreen!30}}{stat_part}"
                    elif approach == 'mpco' or approach.startswith('mpco_'):
                        # MPCO or MPCO variant has best statistic %PI - color it green
                        cell = f"\\cellcolor{{applegreen!30}}{rank_part} & \\cellcolor{{applegreen!30}}{stat_part}"
                    else:
                        # Other approach has best statistic %PI - color it red
                        cell = f"\\cellcolor{{red!20}}{rank_part} & \\cellcolor{{red!20}}{stat_part}"
                else:
                    # No coloring for approaches that don't have the best statistic %PI
                    cell = f"{rank_part} & {stat_part}"
                
                row += f" & {cell}"
            else:
                row += " & & "
        
        latex_content += f"    {row} \\\\\n"
    
    # Calculate average ranks
    avg_ranks = {}
    for approach in approaches:
        ranks = []
        for system in systems:
            if system in results and approach in results[system]:
                ranks.append(results[system][approach]['rank'])
        if ranks:
            avg_ranks[approach] = np.mean(ranks)
    
    # Add average rank row
    avg_row = "    Average $r$"
    for approach in approaches:
        if approach in avg_ranks:
            avg_rank = avg_ranks[approach]
            # Check if this approach has the best average rank
            best_avg_approach = min(avg_ranks.keys(), key=lambda x: avg_ranks[x])
            is_best_avg = approach == best_avg_approach
            
            if is_best_avg:
                avg_row += f" & \\multicolumn{{2}}{{l}}{{\\textbf{{{avg_rank:.2f}}}}}"
            else:
                avg_row += f" & \\multicolumn{{2}}{{l}}{{{avg_rank:.2f}}}"
        else:
            avg_row += " & \\multicolumn{2}{l}{}"
    
    latex_content += f"    \\hline\n    {avg_row}\n"
    
    latex_content += """    \\\\
    \\bottomrule
    \\end{tabular}
    \\end{adjustbox}
    \\end{table}"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    logger.info(f"LaTeX table saved to {output_file}")

def analyze_system_data(df: pd.DataFrame, system_name: str, approaches: List[str], optimization_llms: List[str] = None, use_median: bool = False) -> Dict:
    """
    Analyze data for a specific system.
    
    Args:
        df: DataFrame with evaluation data
        system_name: Name of the system to analyze
        approaches: List of approaches to analyze
        optimization_llms: Optional list of optimization LLMs for each approach (for RQ3)
        use_median: Whether to use median/IQR instead of mean/SD
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing data for {system_name}")
    
    # Calculate PI values for each approach
    approach_pi_values = {}
    for i, approach in enumerate(approaches):
        optimization_llm = optimization_llms[i] if optimization_llms else None
        pi_values = calculate_pi_for_approach(df, system_name, approach, optimization_llm)
        if pi_values:
            approach_pi_values[approach] = pi_values
            if use_median:
                logger.info(f"{approach}: {len(pi_values)} constructs, median PI: {np.median(pi_values):.2f}%")
            else:
                logger.info(f"{approach}: {len(pi_values)} constructs, mean PI: {np.mean(pi_values):.2f}%")
    
    if not approach_pi_values:
        logger.warning(f"No valid data found for {system_name}")
        return {}
    
    # Perform statistical tests
    statistical_results = perform_statistical_tests(approach_pi_values)
    
    # Rank approaches
    ranks = rank_approaches(approach_pi_values, statistical_results)
    
    # Compile results
    results = {}
    for approach in approaches:
        if approach in approach_pi_values:
            pi_values = approach_pi_values[approach]
            if use_median:
                # Calculate median and IQR
                median_pi = np.median(pi_values)
                q1 = np.percentile(pi_values, 25)
                q3 = np.percentile(pi_values, 75)
                iqr_pi = q3 - q1
                results[approach] = {
                    'median_pi': median_pi,
                    'iqr_pi': iqr_pi,
                    'rank': ranks.get(approach, 999),
                    'n_constructs': len(pi_values)
                }
            else:
                # Calculate mean and standard deviation
                results[approach] = {
                    'mean_pi': np.mean(pi_values),
                    'std_pi': np.std(pi_values),
                    'rank': ranks.get(approach, 999),
                    'n_constructs': len(pi_values)
                }
    
    return results

def main():
    """
    Main function to generate RQ LaTeX tables.
    """
    logger.info("Starting RQ LaTeX table generation")
    
    # Configuration: Set to True to use median/IQR, False to use mean/SD
    USE_MEDIAN = False  # Change this to True to use median and IQR
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return
    
    # Define the systems we want to analyze (based on the table template)
    # Map CSV project names to display names
    system_mapping = {
        'BitmapPlusPlus': 'BitmapPlusPlus',
        'llama.cpp': 'Llama.cpp',
        'rpcs3': 'RPCS3',
        'faster-whisper': 'Faster‑Whisper', 
        'Langflow': 'Langflow'
    }
    target_systems = list(system_mapping.keys())  # Use actual project names from CSV
    
    # Process each RQ
    for rq in ['rq1', 'rq2', 'rq3']:
        logger.info(f"\nProcessing {rq.upper()}...")
        
        config = get_rq_config(rq, USE_MEDIAN)
        approaches = config['approaches']
        optimization_llms = config.get('optimization_llms', None)
        
        all_results = {}
        
        # Process each system
        for system in target_systems:
            # Find the CSV file for this system
            system_csv = None
            for csv_file in csv_files:
                # Match the system name in the CSV filename
                if system.lower() in csv_file.name.lower():
                    system_csv = csv_file
                    break
            
            if not system_csv:
                logger.warning(f"No CSV file found for {system}")
                continue
            
            # Load data
            df = load_evaluation_data(str(system_csv))
            if df.empty:
                logger.warning(f"Failed to load data for {system}")
                continue
            
            # Analyze the system
            system_results = analyze_system_data(df, system, approaches, optimization_llms, USE_MEDIAN)
            if system_results:
                all_results[system] = system_results
        
        # Generate LaTeX table
        if all_results:
            generate_latex_table(all_results, rq, USE_MEDIAN)
            logger.info(f"{rq.upper()} LaTeX table generation completed successfully!")
            
            # Print summary
            logger.info(f"\n{rq.upper()} RESULTS SUMMARY")
            logger.info("="*60)
            for system, results in all_results.items():
                logger.info(f"\n{system}:")
                for approach, data in results.items():
                    if USE_MEDIAN:
                        logger.info(f"  {approach}: Rank {data['rank']}, Median PI: {data['median_pi']:.2f}% ± {data['iqr_pi']:.2f}%")
                    else:
                        logger.info(f"  {approach}: Rank {data['rank']}, Mean PI: {data['mean_pi']:.2f}% ± {data['std_pi']:.2f}%")
        else:
            logger.error(f"No valid results found for any system in {rq.upper()}")

if __name__ == "__main__":
    main() 