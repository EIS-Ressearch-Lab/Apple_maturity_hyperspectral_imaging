#!/bin/bash

# Set memory threshold in MB (adjust as needed)
MEM_THRESHOLD=55000  # Minimum free memory required (e.g., 45GB)

# List of Python scripts to run
QUEUE=(
#    8.Predict_and_visualize_Brix_3DCNN.py
#    8.Predict_and_visualize_Firmness_3DCNN.py
#    8.Predict_and_visualize_Starch_3DCNN.py
    # 8.Predict_Test_Hybrid_Brix.py
    # 8.Predict_Test_Hybrid_Firmness.py
    # 8.Predict_Test_Hybrid_Starch.py
    8.Predict_Test_ViT_Brix.py
    8.Predict_Test_ViT_Firmness.py
    8.Predict_Test_ViT_Starch.py
    # 11.Bays_real_code_ViT.py
    # 8.Predict_Test_CNN_Brix.py
    # 8.Predict_Test_CNN_Firmness.py
    # 8.Predict_Test_CNN_Starch.py

)

# Function to check free memory
check_memory() {
    free_mem=$(free -m | awk '/^Mem:/ {print $7}')  # Get available memory in MB
    echo "Available memory: ${free_mem}MB"
    [[ $free_mem -gt $MEM_THRESHOLD ]]  # Returns 0 (true) if memory is enough
}

# Initial wait for 15 minutes
echo "Initial wait for 15 minutes..."
sleep 30 &  # 15 minutes in seconds
initial_wait_pid=$!

# Loop through scripts and run when memory is available
for script in "${QUEUE[@]}"; do
    while ! check_memory; do
        echo "Not enough memory, waiting..."
        sleep 30  # Wait before checking again
    done
    
    echo "Running: $script"
    start_time=$(date +%s)
    start_date=$(date '+%Y-%m-%d %H:%M:%S')
    
    python "$script"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "$start_date,$script,$duration" >> training_log.csv
    echo "Finished: $script (Duration: ${duration}s)"

    wait  # Wait for the script to finish before proceeding
done

echo "All scripts completed!"
