import os
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import torch
import time
import json
import numpy as np
import argparse
import csv
import psutil
import threading
import ast
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetName,
    nvmlShutdown
)

def monitorSystem(batch=-1, interval=1.0, stop_event=None):
    nvmlInit()

    stats = [
        ["timestamp","cpu_percent","node_rss","node_mem_total",
        "gpu0_util", "gpu0_mem","gpu0_totalMem",
        "gpu1_util", "gpu1_mem","gpu1_totalMem",
        "gpu2_util", "gpu2_mem","gpu2_totalMem",
        "gpu3_util", "gpu3_mem","gpu3_totalMem",
        ]
    ]
    
    while not stop_event.is_set():

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        total_rss = sum(proc.memory_info().rss for proc in psutil.process_iter(attrs=['memory_info']))
        system_memory_total = mem.total

        row = [timestamp, cpu_percent,total_rss,system_memory_total]

        for i in range(4):
            handle = nvmlDeviceGetHandleByIndex(i)
            util = nvmlDeviceGetUtilizationRates(handle)
            meminfo = nvmlDeviceGetMemoryInfo(handle)

            row.extend([util.gpu, meminfo.used, meminfo.total])
    
        stats.append(row)
        time.sleep(1)
    
    with open(f"profilingData/batch{batch}_system_stats.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(stats)

def batch_by_char_limit(papers, char_limit, batch_limit, paper_limit):
    """
    Yield batches of (meta, paper) grouped into two lists: one for meta, one for papers.

    Args:
        papers (List[Tuple[meta, paper]]): List of (meta, paper) pairs.
        char_limit (int): Max character count per batch.
        batch_limit (int): Max number of papers per batch.
        paper_limit (int): If a single paper exceeds this limit, yield it alone.

    Yields:
        Tuple[List[meta], List[paper]]: Batched metadata and papers.
    """

    meta_batch = []
    paper_batch = []
    total_chars = 0

    for meta, paper in papers:
        
        paper_len = len(paper)

        # skip papers with no content
        if paper_len == 0:
            continue

        """
        If the paper is bigger than the paper lenth limit, yield it alone. 
        
        Else yield batch if
         * adding another paper exceeds the total character limit
         * adding another paper exceeds the allowed batch size 
        """
        
        if paper_len >= paper_limit:        
            yield [meta], [paper]
            continue
        else:
            if total_chars + paper_len > char_limit or len(paper_batch) + 1 > batch_limit:
                if paper_batch:
                    yield meta_batch, paper_batch
                    meta_batch = []
                    paper_batch = []
                    total_chars = 0
            
            meta_batch.append(meta)
            paper_batch.append(paper)
            total_chars += paper_len

    if paper_batch:
        yield meta_batch, paper_batch

def run_on_gpu(args):
    BASE = "/lus/eagle/projects/radix-io/sockerman/peS2oData/textData/fullTexts/"

    
    t1 = time.time()


    gpu_id, batch_num, info = args
    device = f"cuda:{gpu_id}"
    batch_num += gpu_id
    filepath = BASE + info[0]
    # filepath = "/eagle/projects/radix-io/sockerman/PeS2oEmbeddings/numbers.txt"
    # print(gpu_id, batch_num, info, filepath)
    
     
    
    print(f"[GPU {device}] Processing {filepath} lines {info[1]}-{info[1] + info[2]}")
    
    model_start = time.time()
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-4B",
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "auto",
        },
        tokenizer_kwargs={"padding_side": "left"},
        device=device).to(device)
    model_end = time.time()


    io_start = time.time()
    papers = []
    count = 0
    with open(filepath, "r") as f:
        
        # skip till start
        for _ in range(info[1]):
            next(f)
        
        # get target number of lines
        first = True
        for count in range(info[2]):
            line = next(f, None)
            
            if line is None or line == "":
                break  # end of file reached early

            item = json.loads(line)
            papers.append([item['id'], item['text']])
            # papers.append(line)

            set heuristic parameters based on input type           
            if first:
                first = False  # check the type only once

                # abstracts
                if "s2ag" in item['source']:
                    batch_char_limit = 1000000
                    max_batch_size = 2048
                    paper_len_limit = 100000
                
                # full text
                if "s2orc" in item['source']:
                    batch_char_limit = 150000
                    max_batch_size = 8
                    paper_len_limit = 100000
    

    # print(f"gpu: {gpu_id}",len(papers),papers[0], papers[-1])
    # print()
    return 
    io_stop = time.time()
    embeddings = []
    meta = []
    total = 0
    inf_start  = time.time()
    failed = []

    """
    Aided by a heuristic batch generator, encode the papers in batches when possible. 
    If we hit an OOM, fall back to sequential within that batch.
    """
    with torch.no_grad():
        for meta_batch, batch in batch_by_char_limit(papers, batch_char_limit, max_batch_size, paper_len_limit):

            try:
                e = model.encode(batch)
                embeddings.append(e)
                meta.extend(meta_batch)

                # target = [len(item) for item in batch]
                # total += len(meta_batch)
                # elapsed = time.time() - inf_start
                # rate = total / elapsed
                # print(rate, len(target))
                # print(f"Success Batch stats: ", len(batch), np.sum(target), flush=True)
                # total += len(meta_batch)
            except torch.cuda.OutOfMemoryError:
                
                target = [len(item) for item in batch]
                print(f"GPU {gpu_id} failed batch. Falling back to seq. Batch stats: ", len(batch), np.sum(target), target, flush=True)

                for metaItem, item in zip(meta_batch,batch):
                    try:
                        # total += 1
                        e = model.encode([item])
                        # print(e.shape,flush=True)
                        embeddings.append(e)
                        meta.append(metaItem)
                        # total += 1

                    except torch.cuda.OutOfMemoryError:
                        print(f"GPU {gpu_id} item failed ", len(item), flush=True)
                        failed.append(metaItem)
          

 
    
    inf_end  = time.time()

    print(inf_end-inf_start)
    final_embeddings = np.concatenate(embeddings, axis=0)
    final_meta = np.array(meta)

    new_path = f"/eagle/projects/radix-io/sockerman/peS2oData/embeddings/fullTexts/{info[0].replace(".json","")}-{info[1]}-{info[1]+info[2]}.npz" 
    print(new_path)
    return
    np.savez(new_path, embeddings=final_embeddings, source_pdfs=final_meta)
    
    t2 = time.time()
    
    print(f"[GPU {device}] done with {filepath} lines {info[1]}-{info[1] + info[2]}")
    
    # BASE = "/eagle/projects/radix-io/sockerman/embeddings/perGPULogs/"
    
    with open(BASE + f"batch{batch_id}_gpu{gpu_id}_timing.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_loading', 'i_o', 'inference', 'total'])
        writer.writerow([model_end - model_start, io_stop - io_start, inf_end - inf_start, t2 - t1 ])

    if len(failed) > 0:
        with open(BASE + f"batch{batch_id}_gpu{gpu_id}_failed.csv", "w") as f:
            writer = csv.writer(f)
            f.write(f"source json: {filepath}\n")
            writer.writerow(['keys'])
            f.write("\n".join(failed) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--files", required=True, nargs='+')
    args = parser.parse_args()

    batch_num = int(args.batch)
    files = [ast.literal_eval(item) for item in args.files]

    
 
    # assert len(file_list) == 4, "This script assumes exactly 4 files per node."

    stop_event = threading.Event()
    # t = threading.Thread(target=monitorSystem, kwargs={"stop_event": stop_event, "batch": batch_num})
    # t.start()

    start_time = time.time()
    
    # Zip GPU IDs with the file list
    gpus = len(files)
    args = list(zip(range(gpus), [batch_num,batch_num,batch_num,batch_num], files))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=gpus) as pool:
        pool.map(run_on_gpu, args)

    end_time = time.time()
    stop_event.set()
    # t.join()
    print("Monitoring finished.")

    with open(f"profilingData/batch{batch_num}_runtime.csv", "w") as f:
        f.write(f"runtime\n")
        f.write(f"{end_time - start_time}")
