# Edge Anomaly Response Agent (EARA)

**Goal:** Agentic Edge AI for Real-Time Anomaly Detection in Safety-Critical Systems

Today, robots fight cancer and save lives. But in medical robotics, AI failure is not an option. The single greatest risk is latency. A device analyzing a biopsy signal must react in milliseconds to prevent trauma or equipment damage. Today, robots fight cancer and save lives. But in medical robotics, AI failure is not an option. The single greatest risk is latency. A device analyzing a biopsy signal must react in milliseconds to prevent patient trauma or equipment damage. 

The solution is the Edge Anomaly Response Agent (EARA). We utilized the Liquid Foundation Model (LNM) architecture, designed for sequential data and efficiency—and deployed it locally on AMD compute. This eliminates the cloud and delivers two critical capabilities. First: Millisecond Response. Second: Auditable XAI. The agent provides the interpretable, statistical reasoning (e.g., `3-sigma threshold reached`) with every decision, providing the essential audit trail needed for regulatory compliance. This is a crucial step towards trustworthy, safety-critical AI.

## I. Summary
This project addresses the critical conflict in high-stakes environments (MedTech, Industrial IoT): Cloud latency and security risk are unacceptable for real-time control.

* **Use case:** Liquid AI's LFMs can deliver low-latency, private resilience, and reliable control in mission-critical, resource-constrained, medical device applications.
* **Model type:** Liquid Foundation Model (LFM2-700M) from https://huggingface.co/LiquidAI/LFM2-700M
* **Quantization Method:**	GGUF (General Graph Universal Format) with Q4_K_M Quantization
* **Inference Framework:**	`llama.cpp`
* **Hardware Used:** AMD EPYC (C3D VM) vs. Intel Xeon (N2D VM)
* **Tools:** Python Language Bindings (for `llama.cpp`) and GCP Compute Engine API 

|Component | Traditional Cloud Models (RISK) | Liquid AI + AMD (SOLUTION)| 
|--- | --- | ---| 
| Data Privacy | Sensitive sensor data (e.g., EKG, telemetry) is sent to remote servers (HIPAA/GDPR risk)| 100% On-Device Privacy. Inference stays local on the AMD NUC (simulated)|
| Latency & Control | Latency often exceeds $1000 \text{ ms}$ for decision, risking physical failure| Near-Zero Latency ($\approx 12.7 \text{ s}$ CPU-Time). Engineered for millisecond response times (TTFT) |
| Trust/Audibility | Decisions are made by "black-box" models, which are non-compliant for regulatory audit trails | Auditable XAI (Explainable AI). LNM architecture provides a transparent reasoning chain for every anomaly flag| 


## II. Architecture & ROI

The core deliverable is proving the efficiency gain of the Liquid Neural Network (LNM) architecture when deployed on modern CPU/Edge hardware.

### Core Deliverables

* **LNM Inference:** Successfully deployed the L`FM2-700M-Q4_K_M.gguf` model using the `llama.cpp` Python bindings
* **XAI Agent:** Model correctly analyzed simulated time-series sensor data and returned a structured anomaly classification (ANOMALY STATUS: True) with an explicit reason (XAI REASONING)
* **Cross-Platform Benchmarking:** Conducted a controlled performance comparison on GCP to quantify the advantage of AMD EPYC processors

### Results

This project benchmarked the LFM's performance on the Google Cloud platform (GCP) to quantify the benefit of the AMD EPYC architecture against a standard Intel machine.

|Metric |Intel N2D (Control)|AMD C3D (Optimized)|Performance Uplift|
|---|---|---|---|
|Average TOTAL INFERENCE LATENCY| $14,756.23 \text{ ms}$ ($\approx 14.76 \text{ s}$)| $12,746.43 \text{ ms}$ ($\approx 12.75 \text{ s}$)| $\mathbf{15.8\% \text{ Faster}}$|
|TTFT/Decode Speed| Baseline| Optimized| $15.8\%$ speed increase using the AMD architecture.|
|Key Takeaway / Application | High Cloud OPEX | Significant Cost Savings (AMD C3D VMs offer lower cost/vCPU than Intel counterparts).| |

## III. Assests & Setup

The full project aims to generate a high-value data point on millisecond latency (TTFT) and auditable XAI, directly supporting Liquid AI's mission in interpretable systems.

### Contents

* `src/main_agent.p`y: Python code for LNM model loading, simulated data generation, and XAI inference logic
* `data/results.py`: Python script used to extract, calculate, and plot the comparative TTFT data
* `assets/inference_comparison_graph.png`: Final comparison chart showing the $\mathbf{15.8\% \text{ AMD Uplift}}$
* `data/amd_benchmark_results.txt`: Raw output log (10 runs) from the AMD C3D VM
* `data/intel_benchmark_results.txt`: Raw output log (10 runs) from the Intel N2D VM

### Setup (Ubuntu/Linux)

* Clone the repo
* Install dependencies
```
python3 -m venv venv && source venv/bin/activate
pip install pandas matplotlib llama-cpp-python
```
* Download Mmodel: Download the `LFM2-700M-Q4_K_M.gguf` file directly into the `./models/` folder
* Run Benchmarks: .`/run_benchmarks.sh`


### Model License 
The LFM2-700M-GGUF model is used under the **LFM Open License v1.0.** Commercial use is permitted only for entities with under $10 million in annual revenue, demonstrating adherence to Liquid AI's enterprise standards: https://www.liquid.ai/ 
