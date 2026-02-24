#!/bin/bash

# Bash script to run Python scripts sequentially, checking for GPU and memory availability before each run.
# Set memory threshold in MB (adjust as needed)
# Set initial wait time in seconds (e.g., 2.5 days = 216000 seconds)

# Set memory threshold in MB (adjust as needed)
MEM_THRESHOLD=66000  # Minimum free memory required (e.g., 45GB)

# List of Python scripts to run
QUEUE=(
    # "/media/2tbdisk3/data/Haidee/Python_code_Feb_2025_Aggregated_images/YuanLiu_Code/Code/Bayes_optimisation_files_CNN/11.Bays_optimisation_2D_CNN.py"
    "/media/2tbdisk3/data/Haidee/Python_code_Feb_2025_Aggregated_images/YuanLiu_Code/Code/Bayes_optimisation_files_CNN/3D_CNN.ipynb"
    "/media/2tbdisk3/data/Haidee/Python_code_Feb_2025_Aggregated_images/YuanLiu_Code/Code/Bayes_optimisation_files_CNN/11.Bayes_optimisation_Hybrid.py"
    # "/media/2tbdisk3/data/Haidee/Python_code_Feb_2025_Aggregated_images/YuanLiu_Code/Code/11.Bays_real_code_ViT.py"
    

)

# Check that GPU and memory are available before running each script
wait_for_resources() {
    # echo "Initial 2.5 days wait to allow other processes to finish..."
    # sleep 216000  # 2.5 days in seconds

    while true; do
        free_mem=$(free -m | awk '/^Mem:/ {print $7}')
        gpu_busy=$(nvidia-smi -i 1 | grep -q "python"; echo $?) # Change to 0 or 1 based on GPU index

        if [[ $free_mem -gt $MEM_THRESHOLD && $gpu_busy -ne 0 ]]; then
            return
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') - Resources busy, waiting 1 hour..."
        sleep 3600
    done
}


# Function to check free memory
# check_memory() {
#     free_mem=$(free -m | awk '/^Mem:/ {print $7}')  # Get available memory in MB
#     echo "Available memory: ${free_mem}MB"
#     [[ $free_mem -gt $MEM_THRESHOLD ]]  # Returns 0 (true) if memory is enough
# }

# # Initial wait for 15 minutes
# echo "Initial wait for 15 minutes..."
# sleep 33600 &  # 1 hr in seconds
# initial_wait_pid=$!

# Loop through scripts and run when memory is available
for script in "${QUEUE[@]}"; do
    wait_for_resources  # Wait until resources are available
    python3 "$script"

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
