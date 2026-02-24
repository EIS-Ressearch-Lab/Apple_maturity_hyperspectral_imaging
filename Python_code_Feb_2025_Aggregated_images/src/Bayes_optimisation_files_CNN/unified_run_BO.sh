#!/bin/bash

crash_count=0

while true
do
    python3 unified_main_bo.py
    crash_count=$((crash_count+1))

    # Check if the script has crashed too many times
    if [ $crash_count -gt 5 ]; then
        echo "Too many crashes. Exiting."
        exit 1
    fi

    # Reset crash count if the script runs successfully
    if [ $? -eq 0 ]; then
    crash_count=0
    fi

    echo "Restarting in 5 seconds..."
    sleep 5
done