#!/bin/bash

for i in {11..40}; do
    echo "Checking login$i..."

    # Run the command on the remote server to check GPU memory usage
    # Replace `nvidia-smi` with your command if different
    gpu_usage=$(ssh login$i 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')

    # Define your criteria for low memory usage here. For example:
    low_memory_limit=38000  # 10 GB as a sample limit

    if [[ $gpu_usage -gt $low_memory_limit ]]; then
        echo "Found low memory usage on login$i: $gpu_usage MB free"
        break
    fi
done
