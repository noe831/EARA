#!/usr/bin/env python3
"""Agent for generating simulated sensor data and running anomaly detection via Llama.

Refactored for testability: functions, logging, path validation, and robust JSON extraction.
"""
import os
import time
import logging
import json
import re
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from llama_cpp import Llama

# --- Configuration ---
MODEL_PATH = "models/LFM2-700M-GGUF/LFM2-700M-Q4_K_M-hip-optimized.gguf"
DATA_PATH = "data/simulated_signal.csv"

SYSTEM_PROMPT = (
    "You are an interpretable diagnostic AI agent for medical robotics (LNM architecture)."
    "Analyze the provided time series sensor data for anomalies (e.g. values outside 3-sigma)."
    "Output only a single JSON object. Do not include any text outside of the JSON."
)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


def generate_simulated_data(path: str, rows: int = 100) -> None:
    """Generate simulated sensor CSV and save to `path`. Overwrites existing file."""
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=i) for i in range(rows)]
    sensor_readings = np.random.normal(20, 5, rows)
    # Inject anomalies in last few rows
    anomaly_indices = list(range(max(0, rows - 5), rows))
    for idx in anomaly_indices:
        sensor_readings[idx] = np.random.choice([50, -10])

    df = pd.DataFrame({"timestamp": timestamps, "sensor_reading_1": sensor_readings})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Simulated sensor data saved to %s", path)


def load_and_format_data(file_path: str) -> Tuple[str, Dict[str, str]]:
    """Load CSV and build the user prompt and JSON schema.

    Returns:
        (user_prompt, json_schema)
    """
    df = pd.read_csv(file_path)
    data_text = "Sensor Data Points (Timestamp, Reading):\n"
    data_text += df.to_csv(index=False, header=True)

    json_schema = {
        "anomaly_detected": "boolean (true if reading > 3.0 sigma from mean)",
        "reading_value": "float (the anomalous reading value)",
        "reasoning_chain": "string (explanation of why this reading is considered anomalous)",
        "control_action": "string (suggested action to take based on the anomaly, LOG_FAULT_STATUS or NORMAL)",
    }

    user_prompt = (
        f"Analyze the following sensor data:\n\n{data_text}\n\n"
        f"Identify any anomalies and provide XAI reasoning. "
        f"Output must follow the defined JSON structure: {json.dumps(json_schema)}\n"
    )
    return user_prompt, json_schema


def init_llama(model_path: str, n_ctx: int = 8192) -> Llama:
    logging.info("Initializing Llama model from %s", model_path)
    if not os.path.exists(model_path):
        logging.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=0, verbose=False)
    logging.info("Model loaded")
    return llm


def run_inference(llm: Llama, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Tuple[Dict[str, Any], float]:
    start = time.time()
    output = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    latency_ms = (time.time() - start) * 1000
    return output, latency_ms


def extract_json_from_text(text: str, required_keys: Optional[set] = None) -> Dict[str, Any]:
    """Extract JSON from noisy model output.

    Strategy:
    - Prefer fenced code block containing JSON (```json { ... } ```)
    - Otherwise iterate over JSON-like substrings and parse each
    - If required_keys is provided, return the first parsed object that contains all keys
    - As a last resort, attempt to parse the whole text
    """
    # Try fenced code block
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))

    # Find all non-greedy JSON-like substrings
    for m in re.finditer(r'\{.*?\}', text, re.DOTALL):
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if required_keys:
                if required_keys.issubset(set(parsed.keys())):
                    return parsed
                else:
                    continue
            return parsed
        except Exception:
            continue

    # Last resort
    return json.loads(text)


def main() -> None:
    setup_logging()

    # Step 1: create simulated data
    try:
        generate_simulated_data(DATA_PATH)
    except Exception:
        logging.exception("Failed to generate simulated data")
        return

    if not os.path.exists(DATA_PATH):
        logging.error("Data file missing: %s", DATA_PATH)
        return

    # Step 2: prepare prompt and schema
    try:
        user_prompt, json_schema = load_and_format_data(DATA_PATH)
    except Exception:
        logging.exception("Failed to load or format data from %s", DATA_PATH)
        return

    # Step 3: init model
    try:
        llm = init_llama(MODEL_PATH)
    except FileNotFoundError:
        return
    except Exception:
        logging.exception("Failed to initialize model")
        return

    # Step 4: run inference
    try:
        output, latency_ms = run_inference(llm, SYSTEM_PROMPT, user_prompt)
    except Exception:
        logging.exception("Inference failed")
        return

    response_text = output['choices'][0]['message']['content'].strip()

    # Step 5: extract JSON and validate
    try:
        required_keys = set(json_schema.keys())
        result = extract_json_from_text(response_text, required_keys=required_keys)
        logging.info("--- ERA AGENT DIAGNOSTICS (Edge Result) ---")
        logging.info("ANOMALY STATUS: %s", result.get('anomaly_detected'))
        logging.info("CONTROL ACTION: %s", result.get('control_action'))
        logging.info("XAI REASONING: %s", result.get('reasoning_chain'))
        logging.info("--- PERFORMANCE METRICS ---")
        logging.info("TOTAL INFERENCE LATENCY: %.2f ms", latency_ms)
    except Exception:
        logging.error("Model output did not produce valid JSON or missing keys")
        logging.error("Raw model output was: %s", response_text)
        logging.exception("JSON extraction/parsing error")


if __name__ == '__main__':
    main()
# setup and model loading - download a small, stable model like LFM2-700M-GGUF

# simulate a signal, output it to csv; format timstamp and sensor_reading_1 to the csv file, simulated_signal.csv
# the side of the simulated output will be 100 rows and 2 columns
# the last few rows of the output csv should look like: anamolous values to trigger the model

import time
import pandas as pd
import numpy as np  
from datetime import datetime, timedelta
import json
import re
from llama_cpp import Llama

# --- Configuration ---
# Use the correct path to downloaded GGUF model
MODEL_PATH = "models/LFM2-700M-GGUF/LFM2-700M-Q4_K_M-hip-optimized.gguf"
DATA_PATH = "data/simulated_signal.csv"
# system prompt that frames the model's role
SYSTEM_PROMPT = (
    "You are an interpretable diagnostic AI agent for medical robotics (LNM architecture)."
    "Analyze the provided time series sensor data for anomalies (e.g. values outside 3-sigma)."
    "Output only a single JSON object. Do not include any text outside of the JSON."
    )

# Generate timestamps
start_time = datetime.now()
timestamps = [start_time + timedelta(minutes=i) for i in range(100)]

# Generate sensor readings (normal values)
sensor_readings = np.random.normal(20, 5, 100)  # mean=20, std=5

# Introduce anomalies in the last few rows
anomaly_indices = [95, 96, 97, 98, 99]
for idx in anomaly_indices:
    sensor_readings[idx] = np.random.choice([50, -10])  # Anomalous values

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'sensor_reading_1': sensor_readings
})

# Save to CSV in the data directory
df.to_csv(DATA_PATH, index=False)
print(f"✓ Simulated sensor data saved to {DATA_PATH}")

# Function to load and format data for the model prompt
def load_and_format_data(file_path):
    # Load the simulated time series data (EKG readings or other sensor data)
    df = pd.read_csv(file_path)
    
    # Format the data (e.g., convert timestamps to datetime objects)
    data_text = "Sensor Data Points (Timestamp, Reading):\n"
    data_text += df.to_csv(index=False, header=True)
    
    # Define the output format for the model to follow
    json_schema = {
        "anomaly_detected": "boolean (true if reading > 3.0 sigma from mean)",
        "reading_value": "float (the anomalous reading value)",
        "reasoning_chain": "string (explanation of why this reading is considered anomalous)",
        "control_action": "string (suggested action to take based on the anomaly, LOG_FAULT_STATUS or NORMAL)",
    }

    user_prompt = (
        f"Analyze the following sensor data:\n\n{data_text}\n\n"
        f"Identify any anomalies and provide XAI reasonsing. "
        f"Output must follow the defined JSON structure: {json.dumps(json_schema)}\n"
    )

    return user_prompt

# generate the report
ANOMALY_PROMPT = load_and_format_data(DATA_PATH)

# initialize the Llama model
print("Loading LFM model...")
try:
    llm = Llama(
        model_path=MODEL_PATH, 
        n_ctx=8192,     # context window size (increased from 2048 to handle larger prompts)
        n_gpu_layers=0,  # number of GPU layers to use, set to 0 for CPU only, safe for NUC/Cloud CPU
        verbose=False
    )
    print("✓ Model loaded. Running Inference...")
except FileNotFoundError as e:
    print(f"✗ ERROR: Model file not found at {MODEL_PATH}")
    print(f"  Download the model or update MODEL_PATH to point to your GGUF file.")
    print(f"  Full error: {e}")
    exit(1)
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    exit(1)

# performance measurement
inference_start_time = time.time()

# run the model inference
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANOMALY_PROMPT}
    ],
    max_tokens=512,
    temperature=0.0,
    # capture the output to the first token time
    stream=False
)

end_time = time.time()
# calculate the total latency for XAI reasoning and output
total_latency_ms = (end_time - inference_start_time) * 1000

# extract and validate the output
try: 
    # get the raw text output
    response_text = output['choices'][0]['message']['content'].strip()
    
    # extract JSON from the response (prefer fenced code block JSON)
    # try fenced code block with optional json language, then fall back to first JSON object
    json_str = None
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        json_str = fenced.group(1)
    else:
        # non-greedy match for first JSON object
        obj = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if obj:
            json_str = obj.group(0)

    if json_str:
        XAI_RESULT = json.loads(json_str)
    else:
        # try direct parsing as a last resort (may raise)
        XAI_RESULT = json.loads(response_text)
    
    print( "\n--- ERA AGENT DIAGNOSTICS (Edge Result) ---" )
    print( f"ANOMALY STATUS: {XAI_RESULT.get('anomaly_detected')}" )
    print( f"CONTROL ACTION: {XAI_RESULT.get('control_action')}" )
    print( f"XAI REASONING: {XAI_RESULT.get('reasoning_chain')}") 
    print( "\n--- PERFORMANCE METRICS ---" )
    print( f"TOTAL INFERENCE LATENCY: {total_latency_ms:.2f} ms" )

except (json.JSONDecodeError, KeyError) as e:
    print( f"\nERROR: Model did not produce valid JSON or missing keys:")
    print( f"\nRaw model output was: {response_text}")
    print( str(e) )
