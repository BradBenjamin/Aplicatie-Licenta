import subprocess
import time
import os
from datetime import datetime

# File to store our data
LOG_FILE = "gpu_usage_log.csv"
# How often to check the GPU (in seconds). 600 = 10 minutes.
INTERVAL_SECONDS = 600 

def get_gpu_processes():
    # Asks nvidia-smi for timestamp, process ID, process name, and memory used
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=timestamp,pid,name,used_memory",
        "--format=csv,noheader,nounits"
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().split('\n')
        data = []
        for line in lines:
            if line:
                # Parse the CSV output from nvidia-smi
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 4:
                    data.append({
                        "timestamp": parts[0],
                        "pid": parts[1],
                        "process_name": parts[2],
                        "memory_mb": float(parts[3])
                    })
        return data
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Make sure you are on a node with a GPU.")
        return []

def monitor():
    print(f"Starting GPU monitoring. Logging to {LOG_FILE} every {INTERVAL_SECONDS/60} minutes.")
    write_header = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a") as f:
        if write_header:
            f.write("timestamp,pid,process_name,memory_mb\n")

        while True:
            processes = get_gpu_processes()
            for p in processes:
                # Format: 2026/04/30 13:55:03.123, 1234, python, 4000.0
                f.write(f"{p['timestamp']},{p['pid']},{p['process_name']},{p['memory_mb']}\n")
            f.flush() # Ensure data is written to disk immediately
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    monitor()

# Run using command: nohup python gpu_logger.py > logger_output.log 2>&1 &
# Close with: ps aux | grep gpu_logger.py
# Inspect the log file with: ps aux | grep gpu_log