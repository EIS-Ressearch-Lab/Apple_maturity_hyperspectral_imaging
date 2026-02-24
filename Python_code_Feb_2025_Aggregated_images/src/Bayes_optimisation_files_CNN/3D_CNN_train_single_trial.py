import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import gc
import traceback
import datetime

start_time = datetime.datetime.now()


TEST_MODE = False

# -----------------------------
# Load param file
# -----------------------------
with open(sys.argv[1], "rb") as f:
    early_params = pickle.load(f)

if TEST_MODE:
    epochs = 2
    batch_size = 2
    max_samples = 16
    print(f'Running in test mode: epochs={epochs}, batch_size={batch_size}')
    
else:
    epochs = early_params.get("epochs", 50)
    batch_size = early_params.get("batch_size", 8)
    max_samples = None
    print(f'Running in full mode: epochs={epochs}, batch_size={batch_size}')
    # print("\n===== Loaded Parameters =====")

for k, v in early_params.items():
    print(f"{k}: {v}")
    print("=============================\n")


# -----------------------------
# GPU assignment
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(early_params.get("gpu_id", 0))
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# -----------------------------
# Read arguments
# -----------------------------

PARAM_FILE = sys.argv[1]
RESULT_FILE = sys.argv[2]

# -----------------------------
# Safe exit helper
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
        "filters1": 32,
        "filters2": 64,
        "filters3": 128,
        "filters4": 128,
        "filters5": 64,
        "kernel_size": 3,
        "kernel_size1": 3,
        "dropout": 0.2,
        "pool_size": 2,
    }
    dat_type = "brix"
    safe_exit(score=0.123, status="ok")


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

spectral_bands = 204
input_shape_3d = (spectral_bands, img_size, img_size, 1)
input_shape_cultivar = (6,)

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
# 3D generator
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

                    if data.shape[2] >= spectral_bands:
                        data = data[:, :, :spectral_bands]

                    data = np.transpose(data, (2, 0, 1))

                    if img_size == 40 or img_size == 20:
                        data = data[:, 5:-5, 5:-5]

                    batch_data.append(data)
                    batch_targets.append(targets[offset + i])
                    batch_cult.append(cultivars[offset + i])

                except FileNotFoundError:
                    continue

            if len(batch_data) == 0:
                continue

            batch_data = np.array(batch_data)
            batch_data = np.expand_dims(batch_data, axis=-1)

            batch_targets = np.array(batch_targets)
            batch_cult = np.array(batch_cult)

            yield [batch_data, batch_cult], batch_targets


# -----------------------------
# 3D CNN model
# -----------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D,
    Dense, Input, Add, Activation, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber

def create_model(num_layers, filters1, filters2, filters3, filters4, filters5,
                 kernel_size, kernel_size1, dropout, pool_size):

    spectral_input = Input(shape=input_shape_3d, name="spectral_input")

    # Block 1
    x = Conv3D(filters1, (kernel_size, kernel_size1, kernel_size1), padding="same")(spectral_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_res = Conv3D(filters1, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
    x = Add()([x, x_res])
    x = MaxPooling3D(pool_size=(1, pool_size, pool_size))(x)
    x = Dropout(dropout)(x)

    # Block 2
    x = Conv3D(filters2, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(1, pool_size, pool_size))(x)
    x = Dropout(dropout)(x)

    # Block 3
    if num_layers >= 3:
        x = Conv3D(filters3, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv3D(filters3, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling3D(pool_size=(1, pool_size, pool_size))(x)
        x = Dropout(dropout)(x)

    # Block 4
    if num_layers >= 4:
        x = Conv3D(filters4, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv3D(filters4, (kernel_size, kernel_size1, kernel_size1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling3D(pool_size=(1, pool_size, pool_size))(x)
        x = Dropout(dropout)(x)

    # Block 5
    if num_layers == 5:
        x = Conv3D(filters5, (1, kernel_size1, kernel_size1), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout)(x)

    x = GlobalAveragePooling3D()(x)

    cultivar_input = Input(shape=input_shape_cultivar, name="cultivar_input")
    merged = Concatenate()([x, cultivar_input])

    # Dense head
    x = Dense(512)(merged)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    output = Dense(1)(x)

    model = Model(inputs=[spectral_input, cultivar_input], outputs=output)

    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=Huber(), metrics=["mae"])

    return model


# -----------------------------
# Training
# -----------------------------
try:
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
        "num_layers", "filters1", "filters2", "filters3", "filters4", "filters5",
        "kernel_size", "kernel_size1", "dropout", "pool_size"
    ]

    model_params = {k: params[k] for k in model_keys}


    #### Saftey checks before model creation ####
    max_filter_cap = 128
    max_activation_cap = 200000

    # Extract filters safely
    filters = [
        params.get("filters1", 0),
        params.get("filters2", 0),
        params.get("filters3", 0),
        params.get("filters4", 0),
    ]

    # Hard cap on any single layer
    if any(f > max_filter_cap for f in filters):
        safe_exit(1e9, status="filters_exceed_cap")

    # Optional: cap total feature-map footprint
    estimated_activation_cost = sum(f * f for f in filters)

    if estimated_activation_cost > max_activation_cap:
        safe_exit(1e9, status="activation_cost_too_high")

    # Only now create the model
    try:
        model = create_model(**model_params)
    except RuntimeError as e:
        # Catch CUDA OOM or shape errors
        if "out of memory" in str(e).lower():
            safe_exit(1e9, status="cuda_oom")
        safe_exit(1e9, status=f"model_error:{str(e)[:80]}")

    ###########


    # model = create_model(**model_params)

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
    
    safe_exit(best_loss, status = "ok")

except tf.errors.ResourceExhaustedError:
    safe_exit(1e9, status = "oom")

except Exception:
    traceback.print_exc()
    safe_exit(1e9, status = "crash")
finally:
    gc.collect()
    tf.keras.backend.clear_session()

