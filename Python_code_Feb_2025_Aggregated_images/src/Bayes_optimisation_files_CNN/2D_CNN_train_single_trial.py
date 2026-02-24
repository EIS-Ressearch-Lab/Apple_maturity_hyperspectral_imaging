import os
import sys
# import json
import pickle
import datetime

start_time = datetime.datetime.now()



TEST_MODE = False

with open(sys.argv[1], "rb") as f:
    early_params = pickle.load(f)

if TEST_MODE:
    epochs = 2
    batch_size = 2
    max_samples = 16
    print(f'Running in test mode: epochs={epochs}, batch_size={batch_size}')
    
else:
    epochs = early_params.get("epochs", 50)
    batch_size = early_params.get("batch_size", 32)
    max_samples = None
    print(f'Running in full mode: epochs={epochs}, batch_size={batch_size}')
    # print("\n===== Loaded Parameters =====")

for k, v in early_params.items():
    print(f"{k}: {v}")
    print("=============================\n")




# Assign GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(early_params.get("gpu_id", 0))

import traceback
import numpy as np
import tensorflow as tf
import gc

# -----------------------------
# Thread + GPU safety
# -----------------------------

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# -----------------------------
# Read arguments
# -----------------------------
PARAM_FILE = sys.argv[1]
RESULT_FILE = sys.argv[2]
# TRIED_PARAMS_FILE = "tried_params_2D_CNN.pkl"

# -----------------------------
# Utility: safe exit
# -----------------------------
def safe_exit(score, status):
    global model_params, dat_type

    time_elapsed = datetime.datetime.now() - start_time
        # convert to minutes
    time_elapsed = time_elapsed.total_seconds() / 60
    
    result_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "params": model_params,      # only the hyperparameters
        "data_type": dat_type,
        "score": float(score),
        "status": status,
        "time_elapsed": time_elapsed
    }

    # If file exists, load and append
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "rb") as f:
                existing = pickle.load(f)
            # Ensure it's a list
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    else:
        existing = []

    # Append new entry
    existing.append(result_entry)

    # Save back to file
    with open(RESULT_FILE, "wb") as f:
        pickle.dump(existing, f)

    if status == "ok":
        sys.exit(0)
    elif status == "crash":
        sys.exit(1)
    elif status == "param_load_error":
        sys.exit(2)
    elif status == "oom":
        sys.exit(3)
    elif status == "invalid_config":
        sys.exit(4)
    elif status == "nan_or_inf":
        sys.exit(5)
    elif status == "val_loss_error":
        sys.exit(6)
    else:
        # Unknown status → generic failure code
        sys.exit(7)
 

if os.environ.get("DRY_RUN") == "1":
    print("Dry run mode: skipping training")
    model_params = {
        "num_layers": 2,
        "filters1": 64,
        "filters2": 128,
        "filters3": 256,
        "filters4": 256,
        "filters5": 128,
        "kernel_size": 1,
        "dropout": 0.3,
        "pool_size": 1,
    }
    dat_type = "brix"
    safe_exit(score=0.123, status="ok")


# -----------------------------
# Function to record tried parameters (for analysis/debugging)
# -----------------------------
# def record_tried_params(params, status):
#     # Load existing list
#     if os.path.exists(TRIED_PARAMS_FILE):
#         try:
#             with open(TRIED_PARAMS_FILE, "rb") as f:
#                 tried = pickle.load(f)
#             if not isinstance(tried, list):
#                 tried = []
#         except Exception:
#             tried = []
#     else:
#         tried = []

#     # Append new params
#     tried.append({
#         "params": params,
#         "status": status})

#     # Save back
#     with open(TRIED_PARAMS_FILE, "wb") as f:
#         pickle.dump(tried, f)



# -----------------------------
# Load parameters
# -----------------------------
try:
    with open(PARAM_FILE, "rb") as f:
        params = pickle.load(f)

except Exception:
    safe_exit(1e9, status = "param_load_error")

dat_type = params["dat_type"]
img_size = params["img_size"]

# -----------------------------
# Data loading (from your original script)
# -----------------------------
training_data_path = params["training_data_path"]
validation_file_path = params["validation_file_path"]

def load_data(data_type):
    spectral_path = "/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/"

    if data_type == "brix":
        X_train = np.load(f"{training_data_path}X_train_brix_optimisation_May2025.npy")
        y_train = np.load(f"{training_data_path}Y_train_brix_optimisation_May2025.npy")
        X_val = np.load(f"{validation_file_path}X_validate_all_years_Brix_shuffled.npy")
        y_val = np.load(f"{validation_file_path}Y_validate_all_years_Brix_shuffled.npy")
        train_cult = np.load(f"{training_data_path}X_encoder_brix_optimisation_May2025.npy")
        val_cult = np.load(f"{validation_file_path}X_validate_all_years_Brix_encoder_shuffled.npy")

    elif data_type == "firmness":
        X_train = np.load(f"{training_data_path}X_train_firmness_optimisation_May2025.npy")
        y_train = np.load(f"{training_data_path}Y_train_firmness_optimisation_May2025.npy")
        X_val = np.load(f"{validation_file_path}X_validate_all_years_Firmness_shuffled.npy")
        y_val = np.load(f"{validation_file_path}Y_validate_all_years_Firmness_shuffled.npy")
        train_cult = np.load(f"{training_data_path}X_encoder_firmness_optimisation_May2025.npy")
        val_cult = np.load(f"{validation_file_path}X_validate_all_years_Firmness_encoder_shuffled.npy")

    else:  # starch
        X_train = np.load(f"{training_data_path}X_train_starch_optimisation_May2025.npy")
        y_train = np.load(f"{training_data_path}Y_train_starch_optimisation_May2025.npy")
        X_val = np.load(f"{validation_file_path}X_validate_all_years_Starch_shuffled.npy")
        y_val = np.load(f"{validation_file_path}Y_validate_all_years_Starch_shuffled.npy")
        train_cult = np.load(f"{training_data_path}X_encoder_starch_optimisation_May2025.npy")
        val_cult = np.load(f"{validation_file_path}X_validate_all_years_Starch_encoder_shuffled.npy")

    X_train = [spectral_path + f for f in X_train]
    X_val = [spectral_path + f for f in X_val]

    return X_train, y_train, X_val, y_val, train_cult, val_cult

# -----------------------------
# Generator (from your original script)
# -----------------------------
def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_files = file_list[offset: offset + batch_size]
            batch_data, batch_targets, batch_cult = [], [], []

            for i, file in enumerate(batch_files):
                try:
                    data = np.load(file)
                    if img_size == 40 or img_size == 20:
                        data = data[5:-5, 5:-5, :]
                    batch_data.append(data)
                    batch_targets.append(targets[offset + i])
                    batch_cult.append(cultivars[offset + i])
                except FileNotFoundError:
                    continue

            if len(batch_data) == 0:
                continue

            batch_data = np.array(batch_data)
            batch_targets = np.array(batch_targets)
            batch_cult = np.array(batch_cult)

            expanded = np.repeat(batch_cult[:, None, None, :], img_size, axis=1)
            expanded = np.repeat(expanded, img_size, axis=2)

            combined = np.concatenate([batch_data, expanded], axis=-1)
            yield combined, batch_targets

# -----------------------------
# CNN model (from your original script)
# -----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

input_shape = (img_size, img_size, 210)

def create_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, dropout, pool_size):
    model = Sequential()
    model.add(Conv2D(filters1, (kernel_size, kernel_size), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding="same"))
    model.add(Dropout(dropout))

    model.add(Conv2D(filters2, (kernel_size, kernel_size), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding="same"))
    model.add(Dropout(dropout))

    if num_layers >= 3:
        model.add(Conv2D(filters3, (kernel_size, kernel_size), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding="same"))
        model.add(Dropout(dropout))

    if num_layers >= 4:
        model.add(Conv2D(filters4, (kernel_size, kernel_size), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding="same"))
        model.add(Dropout(dropout))

    if num_layers == 5:
        model.add(Conv2D(filters5, (kernel_size, kernel_size), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding="same"))
        model.add(Dropout(dropout))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1048, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model

# -----------------------------
# Valid configuration check
# -----------------------------
def is_valid_configuration(num_layers, kernel_size):
    h, w, _ = input_shape
    for _ in range(num_layers):
        h = h - kernel_size + 1
        w = w - kernel_size + 1
        if h <= 1 or w <= 1:
            return False
    return True

# -----------------------------
# Run training
# -----------------------------
try:
    if not is_valid_configuration(params["num_layers"], params["kernel_size"]):
        # Exit with a special code
        # record_tried_params(params, status="invalid_config")  # Record the invalid configuration
        safe_exit(1e9, status="invalid_config")

    X_train, y_train, X_val, y_val, train_cult, val_cult = load_data(dat_type)

    if TEST_MODE:
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        train_cult = train_cult[:max_samples]

        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]
        val_cult = val_cult[:max_samples]

    train_gen = data_generator_w_cultivar(X_train, y_train, train_cult, batch_size, img_size)
    val_gen = data_generator_w_cultivar(X_val, y_val, val_cult, batch_size, img_size)

    model_keys = [
    "num_layers",
    "filters1",
    "filters2",
    "filters3",
    "filters4",
    "filters5",
    "kernel_size",
    "dropout",
    "pool_size"
    ]

    model_params = {k: params[k] for k in model_keys}


    model = create_model(**model_params)

    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train)//batch_size,
        validation_data=val_gen,
        validation_steps=len(X_val)//batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=[early]
    )

    best_loss = float(min(history.history["val_loss"]))

    # record_tried_params(model_params, status="ok")  # Record the successful configuration
    safe_exit(best_loss, status="ok")

except tf.errors.ResourceExhaustedError:
    # safe_exit(1e9, "oom")
    # record_tried_params(model_params, status="oom")  # Record the invalid configuration
    safe_exit(1e9, status="oom")

except Exception:
    traceback.print_exc()
    # safe_exit(1e9, "crash")
    # record_tried_params(model_params, status="crash")  # Record the invalid configuration
    safe_exit(1e9, status="crash")

finally:
    gc.collect()
    tf.keras.backend.clear_session()