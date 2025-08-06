import os
import time
import glob
import subprocess
from collections import defaultdict
from collections import OrderedDict
import csv

def read_line_counts_csv(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        data = {filename: int(lines) for filename, lines in reader}

    # Sort by filename for deterministic order
    sorted_data = OrderedDict(sorted(data.items()))
    return sorted_data



# === CONFIG ===
QUEUE_LIMITS = {
    "debug": 1,
    "debug-scaling": 1,
    "preemptable": 5,
}

PBS_TEMPLATE = "template.pbs"

line_counts = read_line_counts_csv("full_text_lc.txt")
TARGET_PER_BATCH = 4000

BATCHES = []             # Each batch is a tuple: (filename, start_line, num_lines, global_offset)
global_line_offset = 0

for filename, total_lines in line_counts.items():
    start = 0
    while start < total_lines:
        remaining = total_lines - start
        take = min(TARGET_PER_BATCH, remaining)

        BATCHES.append((filename, start, take, global_line_offset))

        start += take
        global_line_offset += take

# Optional: sanity check
for item in BATCHES:
    print(item)
print(len(BATCHES))
# exit()

STATE_FILE = "submitted_batches.txt"
SLEEP_SECONDS = 30

# === LOAD STATE ===
submitted = set()
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        for line in f:
            submitted.add(int(line.strip()))

# === MAIN LOOP ===
for batch_id in range(0, len(BATCHES), 4):
    target_batch = BATCHES[batch_id:batch_id+4]
    

    if batch_id in submitted:
        continue  # skip already submitted

    # Wait until one of the queues is under its limit
    while True:
        # Get current job counts by queue
        qstat_out = subprocess.getoutput("qstat -u $USER")
        queue_counts = defaultdict(int)
        for line in qstat_out.splitlines():
            for queue in QUEUE_LIMITS:
                if f"{queue} " in line or (queue[:6] in line and queue == 'preemptable') or (queue[:6] in line and queue == 'debug-scaling'):
                    queue_counts[queue] += 1

        # Find eligible queue
        available_queue = None
        for queue, limit in QUEUE_LIMITS.items():
            if queue_counts[queue] < limit:
                available_queue = queue
                break

        if available_queue:
            break  # Found a free slot

        print(f"[WAIT] All queues full: {dict(queue_counts)}")
        exit()
        # time.sleep(SLEEP_SECONDS)

    # Fill PBS template
    target_batch = [f"\"{str(item)}\"" for item in target_batch]
    batch_files = " ".join(target_batch)
    with open(PBS_TEMPLATE) as f:
        script = f.read()
    
    script = script.replace("__QUEUE__", available_queue)
    script = script.replace("__FILELIST__", batch_files)
    script = script.replace("__JOB__", f"batch{batch_id}")
    script = script.replace("__BATCHNUMBER__", f"{batch_id}")

    job_file = f"batch{batch_id}.sh"
    with open(job_file, "w") as f:
        f.write(script)
    exit()
    subprocess.run(["qsub", job_file])
    print(f"[SUBMIT] Batch {batch_id} → {available_queue}")

    # Save progress
    with open(STATE_FILE, "a") as f:
        f.write(f"{batch_id}\n")
    # exit()
