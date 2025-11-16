import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION: Define the input files ---
# IMPORTANT: Adjust these filenames if you used different names when downloading from the VMs
FILE_PATHS = {
    "AMD EPYC C3D (Test)": "amd_benchmark_results.txt",
    "Intel N2D (Control)": "intel_benchmark_results.txt",
}
OUTPUT_GRAPH_FILE = "comparative_latency_chart.png"
MODEL_NAME = "LFM2-700M-GGUF" 

# --- 2. DATA EXTRACTION FUNCTION ---
def extract_metrics_from_log(file_path):
    """Parses the raw log file using regex to extract latency and status."""
    with open(file_path, 'r') as f:
        log_content = f.read()

    # Regex to find TOTAL INFERENCE LATENCY (ms)
    latency_pattern = r"TOTAL INFERENCE LATENCY: (\d+\.\d+) ms"
    latencies = [float(match) for match in re.findall(latency_pattern, log_content)]

    # Regex to find ANOMALY STATUS (True/False)
    status_pattern = r"ANOMALY STATUS: (True|False)"
    statuses = [status == 'True' for status in re.findall(status_pattern, log_content)]
    
    # Check for consistency (should match the number of runs)
    if len(latencies) != len(statuses):
        print(f"Warning: Latency count ({len(latencies)}) does not match status count ({len(statuses)}) in {file_path}")

    # Return only the necessary data points
    return pd.DataFrame({
        'latency_ms': latencies,
        'anomaly_status': statuses
    })

# --- 3. DATA PROCESSING AND CLEANING ---
all_results = {}
for vm_name, path in FILE_PATHS.items():
    try:
        df_raw = extract_metrics_from_log(path)
        all_results[vm_name] = df_raw
    except FileNotFoundError:
        print(f"ERROR: File not found for {vm_name} at {path}. Skipping this VM.")
        
# Combine all results into one DataFrame for final analysis
final_df = pd.concat(all_results, names=['VM Type', 'Run']).reset_index()

# Calculate key statistics
summary_stats = final_df.groupby('VM Type')['latency_ms'].agg(['mean', 'std', 'count']).reset_index()
summary_stats['mean_sec'] = summary_stats['mean'] / 1000 # Convert to seconds for presentation clarity

# --- 4. VISUALIZATION: Comparative Latency Bar Chart ---
if summary_stats.shape[0] > 1:
    plt.figure(figsize=(10, 6))
    
    # Create the bar chart for average latency
    plt.bar(
        summary_stats['VM Type'], 
        summary_stats['mean_sec'], 
        yerr=summary_stats['std'] / 1000, # Error bars based on standard deviation
        capsize=5, 
        color=['#d62728', '#2ca02c'] # Red for Control, Green for Test/AMD
    )
    
    # Calculate the speed uplift for the pitch
    intel_mean = summary_stats[summary_stats['VM Type'] == 'Intel N2D (Control)']['mean_sec'].iloc[0]
    amd_mean = summary_stats[summary_stats['VM Type'] == 'AMD EPYC C3D (Test)']['mean_sec'].iloc[0]
    speed_uplift = ((intel_mean - amd_mean) / amd_mean) * 100
    
    plt.title(
        f"Inference Speed Uplift: AMD EPYC vs. Intel Baseline\n"
        f"Model: {MODEL_NAME} | AMD C3D is {speed_uplift:.1f}% Faster (Average TTFT)"
    )
    plt.ylabel("Average Inference Latency (Seconds)")
    plt.xlabel("GCP Machine Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the final chart image for your submission assets
    plt.savefig(f"assets_submission/{OUTPUT_GRAPH_FILE}")
    plt.show()

print("\n--- SUMMARY ---")
print(summary_stats[['VM Type', 'mean_sec', 'std']])
print(f"\nFinal Chart Saved: assets_submission/{OUTPUT_GRAPH_FILE}")
