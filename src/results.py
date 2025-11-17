import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

# --- 1. CONFIGURATION ---
INTEL_FILE = "intel_benchmark_results.txt"
AMD_FILE = "amd_benchmark_results.txt"
OUTPUT_IMAGE = "assets_submission/inference_comparison_graph.png"
LATENCY_PATTERN = r"TOTAL INFERENCE LATENCY:\s*(\d+\.\d+)\s*ms"

# --- 2. DATA EXTRACTION FUNCTION ---

def extract_latencies(file_path):
    """Reads a benchmark file and extracts all TOTAL INFERENCE LATENCY values."""
    if not os.path.exists(file_path):
        # In a collaborative environment, we trust the files were uploaded
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, 'r') as f:
        content = f.read()
        # Find all matches using the defined regex pattern
        matches = re.findall(LATENCY_PATTERN, content)
        # Convert captured strings (milliseconds) to floats
        latencies = [float(match) for match in matches]
    return latencies

# --- 3. DATA PROCESSING ---

# Extract data for both VMs
intel_latencies = extract_latencies(INTEL_FILE)
amd_latencies = extract_latencies(AMD_FILE)

# Safely check if we have enough data (must have at least 1 run)
if len(intel_latencies) < 10 or len(amd_latencies) < 10:
    print("WARNING: Less than 10 runs found for one or both platforms. Using all available data.")

if not intel_latencies or not amd_latencies:
    print("FATAL ERROR: Could not extract enough latency data. Check your benchmark files.")
    # Exit gracefully if no data is present
    exit(1)

# Calculate key statistics
avg_intel_latency = np.mean(intel_latencies)
avg_amd_latency = np.mean(amd_latencies)

# The core winning metric: Calculate speed improvement (lower latency = better)
# Uplift (%) = ((Baseline_Avg - Optimized_Avg) / Optimized_Avg) * 100
uplift_percent = ((avg_intel_latency - avg_amd_latency) / avg_amd_latency) * 100

# --- 4. DATA VISUALIZATION (Comparison Bar Chart) ---

# Prepare data for plotting
labels = ['Intel N2D (Baseline)', 'AMD C3D (Optimized)']
averages = [avg_intel_latency, avg_amd_latency]
bar_colors = ['#5A6470', '#800080'] # Gray for control, Purple for Liquid AI focus

plt.style.use('default') 

# Create a clean figure
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(labels, averages, color=bar_colors, width=0.6, capsize=8)

# Format Y-axis to show milliseconds
def ms_formatter(x, pos):
    return f'{int(x):,}'
ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))

# Add latency values on top of the bars (in seconds for better visual impact)
for bar in bars:
    yval = bar.get_height()
    # Display value in seconds
    ax.text(bar.get_x() + bar.get_width()/2, yval + 300, 
            f'{(yval/1000):.2f} s', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set titles and labels for the final presentation slide
ax.set_title(f"LNM Inference Latency: AMD C3D vs. Intel N2D\nAMD Uplift: {uplift_percent:.1f}% Faster Inference", 
             fontsize=16, fontweight='bold')
ax.set_ylabel("Total Inference Latency (ms)", fontsize=12)
ax.set_ylim(0, max(averages) * 1.15) # Adjust Y limit to fit annotations
ax.tick_params(axis='x', labelsize=11) # Clean up axis labels

# Annotate the key winning metric (Uplift) within the plot area
ax.annotate(f"{uplift_percent:.1f}% Faster", 
            xy=(0.5, max(averages) * 0.9), 
            xycoords='data',
            fontsize=16, fontweight='heavy', color='#800080',
            ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the high-resolution image for the 1-slide deck
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
plt.savefig(OUTPUT_IMAGE, dpi=300)

print(f"\n✓ Analysis Complete. Final chart saved to: {OUTPUT_IMAGE}")
print("\n--- KEY WINNING METRIC ---")
print(f"Average Intel Latency (Baseline): {avg_intel_latency:.2f} ms")
print(f"Average AMD Latency (Optimized): {avg_amd_latency:.2f} ms")
print(f"AMD Performance Uplift: {uplift_percent:.1f} % Faster Inference")
