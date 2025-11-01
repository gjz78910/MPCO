#!/usr/bin/env python3
"""
Discussion Analysis Script

This script performs analysis to support the effectiveness of MPCO in the discussion section.
It finds solutions where MPCO provides the most %PI and performs other analyses.

Author: Meta Artemis Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
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

def analyze_mpco_effectiveness(df: pd.DataFrame, project_name: str) -> Dict:
    """
    Analyze MPCO effectiveness for a specific project.
    
    Args:
        df: DataFrame with evaluation data
        project_name: Name of the project
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Analyzing MPCO effectiveness for {project_name}")
    
    original_runtimes = get_original_runtimes(df, project_name)
    if not original_runtimes:
        return {}
    
    # Get all MPCO solutions
    mpco_data = df[
        (df['project_name'] == project_name) & 
        (df['prompt_version'] == 'mpco') &
        (df['construct_id'] != 'All')
    ]
    
    if mpco_data.empty:
        logger.warning(f"No MPCO data found for {project_name}")
        return {}
    
    # Analyze each MPCO solution
    mpco_solutions = []
    for _, row in mpco_data.iterrows():
        optimized_runtimes = parse_runtime_measurements(row['runtime_measurements'])
        if optimized_runtimes:
            pi = calculate_performance_improvement(original_runtimes, optimized_runtimes)
            solution_info = {
                'solution_id': row['solution_id'],
                'solution_name': row.get('solution_name', f"Solution_{row['solution_id']}"),
                'construct_id': row['construct_id'],
                'spec_id': row['spec_id'],
                'spec_name': row['spec_name'],
                'code_optimization_llm': row['code_optimization_llm'],
                'pi': pi,
                'mean_optimized_runtime': np.mean(optimized_runtimes),
                'std_optimized_runtime': np.std(optimized_runtimes),
                'num_measurements': len(optimized_runtimes),
                'avg_runtime': np.mean(optimized_runtimes)  # Added for clarity
            }
            mpco_solutions.append(solution_info)
    
    if not mpco_solutions:
        return {}
    
    # Sort by PI (descending)
    mpco_solutions.sort(key=lambda x: x['pi'], reverse=True)
    
    return {
        'project_name': project_name,
        'original_mean_runtime': np.mean(original_runtimes),
        'original_std_runtime': np.std(original_runtimes),
        'num_original_measurements': len(original_runtimes),
        'mpco_solutions': mpco_solutions,
        'best_mpco_solution': mpco_solutions[0] if mpco_solutions else None,
        'worst_mpco_solution': mpco_solutions[-1] if mpco_solutions else None,
        'avg_mpco_pi': np.mean([s['pi'] for s in mpco_solutions]),
        'max_mpco_pi': max([s['pi'] for s in mpco_solutions]) if mpco_solutions else 0,
        'min_mpco_pi': min([s['pi'] for s in mpco_solutions]) if mpco_solutions else 0,
        'num_mpco_solutions': len(mpco_solutions)
    }

def compare_with_other_approaches(df: pd.DataFrame, project_name: str) -> Dict:
    """
    Compare MPCO with other approaches for a specific project.
    
    Args:
        df: DataFrame with evaluation data
        project_name: Name of the project
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing approaches for {project_name}")
    
    original_runtimes = get_original_runtimes(df, project_name)
    if not original_runtimes:
        return {}
    
    # Define approaches to compare
    approaches = ['mpco', 'chain_of_thought', 'few_shot', 'contextual_prompting', 'baseline']
    
    approach_results = {}
    
    for approach in approaches:
        approach_data = df[
            (df['project_name'] == project_name) & 
            (df['prompt_version'] == approach) &
            (df['construct_id'] != 'All')
        ]
        
        if approach_data.empty:
            continue
        
        pi_values = []
        for _, row in approach_data.iterrows():
            optimized_runtimes = parse_runtime_measurements(row['runtime_measurements'])
            if optimized_runtimes:
                pi = calculate_performance_improvement(original_runtimes, optimized_runtimes)
                pi_values.append(pi)
        
        if pi_values:
            approach_results[approach] = {
                'mean_pi': np.mean(pi_values),
                'std_pi': np.std(pi_values),
                'max_pi': max(pi_values),
                'min_pi': min(pi_values),
                'num_solutions': len(pi_values),
                'pi_values': pi_values
            }
    
    return {
        'project_name': project_name,
        'approach_results': approach_results
    }

def find_top_mpco_solutions_across_systems(top_n: int = 50) -> List[Dict]:
    """
    Find the top N MPCO solutions across all systems.
    
    Args:
        top_n: Number of top solutions to return (default: 50)
        
    Returns:
        List of top MPCO solutions across all systems
    """
    logger.info(f"Finding top {top_n} MPCO solutions across all systems")
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return []
    
    all_solutions = []
    
    for csv_file in csv_files:
        # Extract project name from filename
        project_name = csv_file.stem.split('_')[2]  # evaluation_data_PROJECTNAME_...
        
        # Load data
        df = load_evaluation_data(str(csv_file))
        if df.empty:
            continue
        
        # Analyze MPCO effectiveness
        analysis = analyze_mpco_effectiveness(df, project_name)
        if analysis and analysis.get('mpco_solutions'):
            for solution in analysis['mpco_solutions']:
                solution['project_name'] = project_name
                all_solutions.append(solution)
    
    # Sort by PI (descending) and return top N
    all_solutions.sort(key=lambda x: x['pi'], reverse=True)
    return all_solutions[:top_n]

def analyze_mpco_consistency() -> Dict:
    """
    Analyze the consistency of MPCO performance across different constructs.
    
    Returns:
        Dictionary with consistency analysis
    """
    logger.info("Analyzing MPCO consistency across constructs")
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return {}
    
    all_mpco_pi_values = []
    project_consistency = {}
    
    for csv_file in csv_files:
        project_name = csv_file.stem.split('_')[2]
        
        # Load data
        df = load_evaluation_data(str(csv_file))
        if df.empty:
            continue
        
        # Get MPCO PI values for this project
        analysis = analyze_mpco_effectiveness(df, project_name)
        if analysis and analysis.get('mpco_solutions'):
            pi_values = [s['pi'] for s in analysis['mpco_solutions']]
            all_mpco_pi_values.extend(pi_values)
            
            project_consistency[project_name] = {
                'mean_pi': np.mean(pi_values),
                'std_pi': np.std(pi_values),
                'cv_pi': np.std(pi_values) / np.mean(pi_values) if np.mean(pi_values) != 0 else 0,
                'num_solutions': len(pi_values),
                'pi_values': pi_values
            }
    
    if not all_mpco_pi_values:
        return {}
    
    return {
        'overall_mpco_performance': {
            'mean_pi': np.mean(all_mpco_pi_values),
            'std_pi': np.std(all_mpco_pi_values),
            'cv_pi': np.std(all_mpco_pi_values) / np.mean(all_mpco_pi_values) if np.mean(all_mpco_pi_values) != 0 else 0,
            'max_pi': max(all_mpco_pi_values),
            'min_pi': min(all_mpco_pi_values),
            'total_solutions': len(all_mpco_pi_values)
        },
        'project_consistency': project_consistency
    }

def calculate_total_experiment_time() -> Dict:
    """
    Calculate total time spent on all experiments.
    
    Returns:
        Dictionary with timing information
    """
    logger.info("Calculating total experiment time")
    
    # Find CSV files in the results directory
    results_dir = Path("results")
    csv_files = list(results_dir.glob("evaluation_data_*.csv"))
    
    if not csv_files:
        logger.error("No evaluation data CSV files found in results directory")
        return {}
    
    total_measurements = 0
    total_runtime = 0
    project_times = {}
    
    for csv_file in csv_files:
        project_name = csv_file.stem.split('_')[2]
        
        # Load data
        df = load_evaluation_data(str(csv_file))
        if df.empty:
            continue
        
        project_total_runtime = 0
        project_measurements = 0
        
        for _, row in df.iterrows():
            runtime_str = row.get('runtime_measurements', '')
            if runtime_str:
                measurements = parse_runtime_measurements(runtime_str)
                if measurements:
                    project_total_runtime += sum(measurements)
                    project_measurements += len(measurements)
                    total_runtime += sum(measurements)
                    total_measurements += len(measurements)
        
        if project_measurements > 0:
            project_times[project_name] = {
                'total_runtime': project_total_runtime,
                'num_measurements': project_measurements,
                'avg_runtime': project_total_runtime / project_measurements
            }
    
    return {
        'total_runtime': total_runtime,
        'total_measurements': total_measurements,
        'avg_runtime_per_measurement': total_runtime / total_measurements if total_measurements > 0 else 0,
        'project_times': project_times
    }

def generate_discussion_insights() -> Dict:
    """
    Generate insights for the discussion section.
    
    Returns:
        Dictionary with discussion insights
    """
    logger.info("Generating discussion insights")
    
    # Find top MPCO solutions
    top_solutions = find_top_mpco_solutions_across_systems(50)
    
    # Analyze consistency
    consistency_analysis = analyze_mpco_consistency()
    
    # Calculate total experiment time
    timing_analysis = calculate_total_experiment_time()
    
    # Generate insights
    insights = {
        'top_mpco_solutions': top_solutions,
        'consistency_analysis': consistency_analysis,
        'timing_analysis': timing_analysis,
        'key_findings': {}
    }
    
    if top_solutions:
        # Best performing MPCO solution
        best_overall = top_solutions[0]
        insights['key_findings']['best_performing_solution'] = {
            'project': best_overall['project_name'],
            'pi': best_overall['pi'],
            'solution_id': best_overall['solution_id'],
            'solution_name': best_overall['solution_name'],
            'avg_runtime': best_overall['avg_runtime'],
            'construct': best_overall['construct_id'],
            'spec': best_overall['spec_name'],
            'llm': best_overall['code_optimization_llm']
        }
        
        # Average performance across top solutions
        avg_pi = np.mean([s['pi'] for s in top_solutions])
        insights['key_findings']['average_performance'] = avg_pi
        
        # Range of performance
        pi_values = [s['pi'] for s in top_solutions]
        insights['key_findings']['performance_range'] = {
            'min': min(pi_values),
            'max': max(pi_values),
            'std': np.std(pi_values)
        }
    
    if consistency_analysis and consistency_analysis.get('overall_mpco_performance'):
        overall = consistency_analysis['overall_mpco_performance']
        insights['key_findings']['consistency'] = {
            'coefficient_of_variation': overall['cv_pi'],
            'mean_performance': overall['mean_pi'],
            'std_performance': overall['std_pi']
        }
    
    return insights

def print_discussion_insights(insights: Dict):
    """
    Print formatted discussion insights.
    
    Args:
        insights: Dictionary with discussion insights
    """
    print("\n" + "="*80)
    print("DISCUSSION ANALYSIS INSIGHTS")
    print("="*80)
    
    if insights.get('key_findings'):
        findings = insights['key_findings']
        
        if 'best_performing_solution' in findings:
            best = findings['best_performing_solution']
            print(f"\n🏆 BEST PERFORMING MPCO SOLUTION:")
            print(f"   Project: {best['project']}")
            print(f"   Performance Improvement: {best['pi']:.2f}%")
            print(f"   Solution ID: {best['solution_id']}")
            print(f"   Solution Name: {best['solution_name']}")
            print(f"   Average Runtime: {best['avg_runtime']:.4f}s")
            print(f"   Construct: {best['construct']}")
            print(f"   Specification: {best['spec']}")
            print(f"   Optimization LLM: {best['llm']}")
        
        if 'average_performance' in findings:
            print(f"\n📊 AVERAGE MPCO PERFORMANCE:")
            print(f"   Across top 50 solutions: {findings['average_performance']:.2f}%")
        
        if 'performance_range' in findings:
            range_info = findings['performance_range']
            print(f"\n📈 PERFORMANCE RANGE:")
            print(f"   Minimum: {range_info['min']:.2f}%")
            print(f"   Maximum: {range_info['max']:.2f}%")
            print(f"   Standard Deviation: {range_info['std']:.2f}%")
        
        if 'consistency' in findings:
            consistency = findings['consistency']
            print(f"\n🔄 CONSISTENCY ANALYSIS:")
            print(f"   Coefficient of Variation: {consistency['coefficient_of_variation']:.3f}")
            print(f"   Mean Performance: {consistency['mean_performance']:.2f}%")
            print(f"   Standard Deviation: {consistency['std_performance']:.2f}%")
    
    if insights.get('timing_analysis'):
        timing = insights['timing_analysis']
        print(f"\n⏱️  EXPERIMENT TIMING ANALYSIS:")
        print(f"   Total Runtime: {timing['total_runtime']:.2f} seconds")
        print(f"   Total Measurements: {timing['total_measurements']:,}")
        print(f"   Average Runtime per Measurement: {timing['avg_runtime_per_measurement']:.4f} seconds")
        
        if timing.get('project_times'):
            print(f"\n📊 RUNTIME BY PROJECT:")
            for project, time_info in timing['project_times'].items():
                print(f"   {project}: {time_info['total_runtime']:.2f}s "
                      f"({time_info['num_measurements']:,} measurements, "
                      f"avg: {time_info['avg_runtime']:.4f}s)")
    
    if insights.get('top_mpco_solutions'):
        print(f"\n📋 TOP 50 MPCO SOLUTIONS BY %PI:")
        print(f"{'Rank':<4} {'Project':<15} {'PI%':<8} {'Solution ID':<12} {'LLM':<15} {'Avg Runtime':<12} {'Construct':<10}")
        print("-" * 80)
        for i, solution in enumerate(insights['top_mpco_solutions'][:50]):
            print(f"{i+1:<4} {solution['project_name']:<15} {solution['pi']:<8.2f} "
                  f"{solution['solution_id']:<12} {solution['code_optimization_llm']:<15} "
                  f"{solution['avg_runtime']:<12.4f} {solution['construct_id']:<10}")

def main():
    """
    Main function to run discussion analysis.
    """
    logger.info("Starting discussion analysis")
    
    # Generate insights
    insights = generate_discussion_insights()
    
    # Print insights
    print_discussion_insights(insights)
    
    # Save detailed results to file
    output_file = "results/discussion_analysis_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DISCUSSION ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        if insights.get('timing_analysis'):
            f.write("EXPERIMENT TIMING ANALYSIS:\n")
            f.write("-"*30 + "\n")
            timing = insights['timing_analysis']
            f.write(f"Total Runtime: {timing['total_runtime']:.2f} seconds\n")
            f.write(f"Total Measurements: {timing['total_measurements']:,}\n")
            f.write(f"Average Runtime per Measurement: {timing['avg_runtime_per_measurement']:.4f} seconds\n\n")
            
            if timing.get('project_times'):
                f.write("Runtime by Project:\n")
                for project, time_info in timing['project_times'].items():
                    f.write(f"  {project}: {time_info['total_runtime']:.2f}s "
                           f"({time_info['num_measurements']:,} measurements, "
                           f"avg: {time_info['avg_runtime']:.4f}s)\n")
                f.write("\n")
        
        if insights.get('top_mpco_solutions'):
            f.write("TOP 50 MPCO SOLUTIONS BY %PI:\n")
            f.write("-"*30 + "\n")
            for i, solution in enumerate(insights['top_mpco_solutions'][:50]):
                f.write(f"{i+1:2d}. {solution['project_name']:<15} {solution['pi']:6.2f}% "
                       f"ID:{solution['solution_id']:<8} {solution['code_optimization_llm']:<15} "
                       f"Runtime:{solution['avg_runtime']:8.4f}s {solution['construct_id']}\n")
            f.write("\n")
        
        if insights.get('consistency_analysis'):
            f.write("CONSISTENCY ANALYSIS:\n")
            f.write("-"*20 + "\n")
            overall = insights['consistency_analysis']['overall_mpco_performance']
            f.write(f"Overall MPCO Performance:\n")
            f.write(f"  Mean PI: {overall['mean_pi']:.2f}%\n")
            f.write(f"  Std PI: {overall['std_pi']:.2f}%\n")
            f.write(f"  Coefficient of Variation: {overall['cv_pi']:.3f}\n")
            f.write(f"  Max PI: {overall['max_pi']:.2f}%\n")
            f.write(f"  Min PI: {overall['min_pi']:.2f}%\n")
            f.write(f"  Total Solutions: {overall['total_solutions']}\n\n")
    
    logger.info(f"Discussion analysis completed. Results saved to {output_file}")

if __name__ == "__main__":
    main() 