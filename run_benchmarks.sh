#!/bin/bash

LOG_FILE="benchmark_results.txt"
SCRIPT="./hackathon_venv/bin/python3 src/main_agent.py"

# Clear previous runs
echo "--- Starting Benchmark Runs: $(date) ---" > $LOG_FILE
echo "Running 10 consecutive tests. Results saved to $LOG_FILE"

# Loop 10 times to run the test
for i in {1..10}
do
    echo "--- RUN $i ---" >> $LOG_FILE
    # Execute the Python script and append the output to the log file
    $SCRIPT 2>&1 >> $LOG_FILE 
    echo "---------------------------------" >> $LOG_FILE
    sleep 1 # Wait one second between runs
done

echo "--- Benchmarks Complete ---" >> $LOG_FILE
cat $LOG_FILE # Display the results immediately
