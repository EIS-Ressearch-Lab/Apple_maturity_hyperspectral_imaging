import os
import pickle
import subprocess
import uuid
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
import time
import csv

TEST_MODE = False

# ============================================================
# Global settings
# ============================================================
today = date.today().strftime('%Y-%m-%d')
run_id = "01"
img_size = 40

training_data_path = "/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/"
validation_file_path = "/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/"
code_path = "/media/2tbdisk3/data/Haidee/Python_code_Feb_2025_Aggregated_images/YuanLiu_Code/Code/"


TIMEOUT_SECONDS = 7200

if TEST_MODE:
    N_CALLS = 2
else:
    N_CALLS = 100

MAX_PARALLEL = 2
GPU_POOL = [0, 1]

# # Log file
# LOG_FILE = f"{training_data_path}{today}_BO_run_log.csv"

# # Create log file with header if it doesn't exist
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "timestamp",
#             "model",
#             "dat_type",
#             "trial_index",
#             "score",
#             "gpu_id",
#             "seconds_elapsed"
#         ])



# ============================================================
# Enable / disable models here
# ============================================================
ENABLED_MODELS = {
    # "2D_CNN": True,
    "hybrid": True,
    "ViT": True,
    "3D_CNN": True,
}



# ============================================================
# Search spaces for each model
# ============================================================
SEARCH_SPACES = {

    "2D_CNN": [
        Integer(1, 5, name="num_layers"),
        Integer(32, 124, name="filters1"),
        Integer(32, 256, name="filters2"),
        Integer(32, 512, name="filters3"),
        Integer(32, 256, name="filters4"),
        Integer(32, 128, name="filters5"),
        Integer(1, 2, name="kernel_size"),
        Real(0.1, 0.5, name="dropout"),
        Integer(1, 2, name="pool_size"),
    ],

    "3D_CNN": [
        Integer(1, 4, name="num_layers"),
        Integer(8, 32, name="filters1"),
        Integer(16, 64, name="filters2"),
        Integer(16, 64, name="filters3"),
        Integer(16, 48, name="filters4"),
        Integer(16, 32, name="filters5"),
        Integer(3, 5, name="kernel_size"),
        Integer(2, 3, name="kernel_size1"),
        Real(0.1, 0.4, name="dropout"),
        Integer(2, 3, name="pool_size"),
    ],

    "hybrid": [
        Integer(1, 3, name="CNN_layers"),
        Integer(8, 64, name="filter1"),
        Integer(8, 96, name="filter2"),
        Integer(32, 128, name="filter3"),
        Integer(2, 5, name="kernel"),
        Real(0.1, 0.4, name="dropout"),
        Integer(2, 3, name="pool_size"),
        Integer(1, 5, name="transformer_layers"),
        Integer(2, 10, name="patch_size"),
        Categorical([32, 64, 96, 128, 160, 192, 224, 256], name="projection_dim"),
        Categorical([1, 2, 4, 8], name="num_heads"),
        # Categorical([(128, 64), (256, 128, 64), (256, 128)], name="mlp_head_units"),
        Categorical(["128-64", "256-128-64", "256-128"], name="mlp_head_units"),
    ],

    "ViT": [
        Integer(2, 10, name="patch_size"),
        Integer(64, 192, name="projection_dim"),
        Integer(1, 5, name="transformer_layers"),
        Categorical([1, 2, 4, 8], name="num_heads"),
        # Categorical([(128, 64), (256, 128, 64), (256, 128)], name="mlp_head_units"),
        Categorical(["128-64", "256-128-64", "256-128"], name="mlp_head_units"),
        Real(0.1, 0.5, name="dropout_rate"),
    ],
}


# ============================================================
# Train_single filenames for each model
# ============================================================
TRAIN_FILES = {
    "2D_CNN": "Bayes_optimisation_files_CNN/2D_CNN_train_single_trial.py",
    "3D_CNN": "Bayes_optimisation_files_CNN/3D_CNN_train_single_trial.py",
    "hybrid": "Bayes_optimisation_files_CNN/hybrid_train_single_trial.py",
    "ViT": "Bayes_optimisation_files_CNN/ViT_train_single_trial.py",
}

# ============================================================
# Define a function to check if the configuration has been attempted
# ============================================================


def already_tried_params_per_feature(params_dict, tried_params, dat_type=None):

    # Extract only hyperparameters (exclude metadata)
    hyperparam_keys = set(params_dict.keys()) - {
        "dat_type", "img_size",
        "training_data_path", "validation_file_path", "gpu_id"
    }

    current_hyperparams = {
        k: params_dict[k] for k in hyperparam_keys
    }

    for entry in tried_params:

        if not isinstance(entry, dict):
            continue

        if entry.get("data_type") == dat_type:
            if entry.get("params") == current_hyperparams:
                return True

    return False




# ============================================================
# Run a single trial
# ============================================================
def run_trial(model_name, params_dict, dat_type, tried_params):

    # -----------------------------------------
    # Load previously tried params
    # -----------------------------------------
    tried = already_tried_params_per_feature(params_dict, tried_params, dat_type)

    if tried: # If true, is skipped, if false, is new parameter set for model x feature type
        print("Skipping previously tried parameters")
        return None, params_dict   # skip trial, does NOT count as a trial

    # -----------------------------------------
    # Create temp param + result files
    # -----------------------------------------
    param_file = f"params_{uuid.uuid4().hex}.pkl" # Unique file that the worker .py file reads from
    result_file = f"result_{uuid.uuid4().hex}.pkl" # The unique file that the worker .py file saves to

    with open(param_file, "wb") as f:
        pickle.dump(params_dict, f)
    
    # -----------------------------------------
    # Run worker
    # -----------------------------------------
    try:
        subprocess.run(
            ["python3", os.path.join(code_path, TRAIN_FILES[model_name]), param_file, result_file],
            timeout=TIMEOUT_SECONDS,
            check=False
        )
    except subprocess.TimeoutExpired:
        return None, params_dict # Skip trial if it times out (treat as if it was never attempted)

    if not os.path.exists(result_file):
        return None, params_dict # Skip if no result file (e.g. worker crashed without writing result)

    with open(result_file, "rb") as f:
        result = pickle.load(f)

    if isinstance(result, list) and len(result) > 0:
        result_entry = result[-1]
    elif isinstance(result, dict):
        result_entry = result
    else:
        return None, params_dict

    score = result_entry.get("score", None)

    persistent_file = f"all_results_{model_name}_{dat_type}.pkl"

    # Load existing results
    if os.path.exists(persistent_file):
        try:
            with open(persistent_file, "rb") as f:
                all_results = pickle.load(f)
            if not isinstance(all_results, list):
                all_results = [all_results]
        except Exception:
            all_results = []
    else:
        all_results = []

    # Append new entry
    all_results.append(result_entry)

    # Save back
    with open(persistent_file, "wb") as f:
        pickle.dump(all_results, f)

    # Clean up temp files
    os.remove(param_file)
    os.remove(result_file)
    
    return score, result_entry


# ============================================================
# Main BO loop for each model + datatype
# ============================================================
for model_name in ["2D_CNN", "3D_CNN", "hybrid", "ViT"]:

    # Skip if model is disabled
    if not ENABLED_MODELS.get(model_name, False):
        print(f"\n>>> Skipping {model_name} (disabled)")
        continue

    print(f"\n==============================")
    print(f"=== Running BO for {model_name} ===")
    print(f"==============================")

    search_space = SEARCH_SPACES[model_name]

    for dat_type in ["brix", "firmness", "starch"]:
        print(f"\n--- Dataset: {dat_type} ---")

        checkpoint_path = (
            f"{training_data_path}Bayes_opt_2026/{model_name}_{run_id}_BO_checkpoint_{dat_type}.pkl"
        )

        data_save_path = f"all_results_{model_name}_{dat_type}.pkl"

        # Resume or start fresh
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            optimizer = data["optimizer"]
            tried_params = data["params"]
            tried_scores = data["scores"]
            # print(f"Resuming from {len(tried_params)} trials.")
            assert all(
                isinstance(x, dict) and
                "params" in x and
                "data_type" in x
                for x in tried_params
            )
        else:
            optimizer = Optimizer(search_space, random_state=779)
            tried_params, tried_scores = [], []

        if os.path.exists(data_save_path):
            with open(data_save_path, "rb") as f:
                all_res = pickle.load(f)
            num_ok = sum(entry["status"] == "ok" for entry in all_res)

            if num_ok >= N_CALLS:
                print(f"Already have {num_ok} successful trials for {model_name} on {dat_type}, skipping BO.")
                continue
            else:
                print(f"Resuming from {num_ok} successful trials for {model_name} on {dat_type}.")

        oom_streak = {gpu: 0 for gpu in GPU_POOL} # Per-gpu OOM streaks

                # Parallel execution
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:

            futures = {}

            # Launch initial batch (fill all GPUs immediately)
            for i, gpu_id in enumerate(GPU_POOL[:MAX_PARALLEL]):
                next_params = optimizer.ask()
                param_dict = dict(zip([d.name for d in search_space], next_params))

                param_dict["dat_type"] = dat_type
                param_dict["img_size"] = img_size
                param_dict["training_data_path"] = training_data_path
                param_dict["validation_file_path"] = validation_file_path
                param_dict["gpu_id"] = gpu_id #GPU_POOL[i % len(GPU_POOL)]

                future = executor.submit(run_trial, model_name, param_dict, dat_type, tried_params)
                futures[future] = (next_params, param_dict)

            # Process completed futures and launch new ones

            valid_trials = num_ok

            pbar = tqdm(total=N_CALLS, initial=valid_trials,
                        desc=f"{model_name} ({dat_type})")


            while valid_trials < N_CALLS and futures:
            
                # Wait for ANY worker to finish
                for future in as_completed(futures):
                    params_used, meta = futures[future]
                    score, result_entry = future.result()

                    del futures[future]  # Free memory immediately

                    # Extract status and GPU used
                    status = result_entry.get("status") if result_entry else "crash"
                    gpu_id = meta.get("gpu_id", None)

                    # Track OOM streak per gpu
                    if status == "oom" and gpu_id is not None:
                        oom_streak[gpu_id] += 1
                    elif gpu_id is not None:
                        oom_streak[gpu_id] = 0

                    # If too many OOMs in a row → reset only the affected GPU
                    if gpu_id is not None and oom_streak[gpu_id] >= 5:
                        print(f"⚠️ 5 OOM errors in a row on GPU {gpu_id} — resetting and retrying...")

                        # Kill all running futures
                        for f in futures:
                            f.cancel()
                        futures.clear()

                        os.system(f"nvidia-smi --gpu-reset -i {gpu_id}")
                        # Reset streak
                        oom_streak[gpu_id] = 0

                        # Skip BO update and retry
                        break # Important: break so we immediately retry instead of waiting for more futures to complete, reenter while loop

                    # Normal sucessful trial

                    if score is not None and result_entry.get("status") == "ok":
                        print(f"✅ Trial completed on GPU {gpu_id} with score {score}")
                        optimizer.tell(params_used, score)

                        hyperparams_only = {
                            k: v for k, v in meta.items()
                            if k in [d.name for d in search_space]
                        }

                        tried_params.append({
                            "params": hyperparams_only,
                            "data_type": dat_type
                        })
                        tried_scores.append(score)

                        valid_trials += 1
                        pbar.update(1)

                        with open(checkpoint_path, "wb") as f:
                            pickle.dump({
                                    "optimizer": optimizer,
                                    "params": tried_params,
                                    "scores": tried_scores
                                }, f)
                        print(f"Checkpoint saved with {valid_trials} valid trials.")


                    # Lauch next trial on same GPU that just freed up
                    if valid_trials < N_CALLS and gpu_id is not None:
                        next_params = optimizer.ask()
                        param_dict = dict(zip([d.name for d in search_space], next_params))

                        param_dict["dat_type"] = dat_type
                        param_dict["img_size"] = img_size
                        param_dict["training_data_path"] = training_data_path
                        param_dict["validation_file_path"] = validation_file_path
                        param_dict["gpu_id"] = gpu_id

                        new_future = executor.submit(run_trial, model_name, param_dict, dat_type, tried_params)
                        futures[new_future] = (next_params, param_dict)

                    break # Handle one completed future at a time (important for OOM handling and immediate retries)

            pbar.close()

                
            # # Finish remaining futures
            # for future in as_completed(futures):
            #     params_used, meta = futures[future]
            #     score, _ = future.result()

            #     if score is not None:
            #         optimizer.tell(params_used, score)
            #     else:
            #         continue

            #     hyperparams_only = {
            #         k: v for k, v in meta.items()
            #         if k in [d.name for d in search_space]
            #     }


            #     tried_params.append({
            #         "params": hyperparams_only,      # full dict with metadata
            #         "data_type": dat_type
            #     })
            #     tried_scores.append(score)

            #     with open(checkpoint_path, "wb") as f:
            #         pickle.dump({
            #             "optimizer": optimizer,
            #             "params": tried_params,
            #             "scores": tried_scores
            #         }, f)
                
            #     # with open(f"tried_params_{model_name}.pkl", "wb") as f:
            #     #     pickle.dump(tried_params, f)  

            #     print(f"Score={score}")