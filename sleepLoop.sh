#!/bin/bash

WATCH_DIR="/eagle/projects/radix-io/sockerman/Bioarxiv.nougat/embeddings/"

while [ $(find "$WATCH_DIR" -type f | wc -l) -lt 512 ]; do
    echo "[$(date)] Running Python script..."
    python3 queue_watch.py

    # echo "Sleeping for 10 minutes..."
    # sleep 600
    echo "Sleeping for 5 minutes..."
    sleep 300
done

echo "[$(date)] Found 512 or more files in $WATCH_DIR. Done."
