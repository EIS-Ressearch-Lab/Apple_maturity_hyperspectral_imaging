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

# ============================================================
# Early param load (for GPU assignment)
# ============================================================
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



os.environ["CUDA_VISIBLE_DEVICES"] = str(early_params.get("gpu_id", 0))
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})



# ============================================================
# CLI args
# ============================================================
PARAM_FILE = sys.argv[1]
RESULT_FILE = sys.argv[2]

# ============================================================
# Safe exit helper
# ============================================================
def safe_exit(score, status):
    global model_params, dat_type

    time_elapsed = datetime.datetime.now() - start_time
        # convert to minutes
    time_elapsed = time_elapsed.total_seconds() / 60

    clean_params = {}
    for k, v in model_params.items():
        if isinstance(v, tuple):
            clean_params[k] = "-".join(str(x) for x in v)
        else:
            clean_params[k] = v
    
    result_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "params": clean_params,      # only the hyperparameters
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
        "patch_size": 4,
        "projection_dim": 128,
        "transformer_layers": 2,
        "num_heads": 2,
        "mlp_head_units": "128-64", # or tuple (128, 64),
        "dropout_rate": 0.2,
    }
    dat_type = "brix"
    safe_exit(score=0.123, status="ok")


# ============================================================
# Full params load
# ============================================================
try:
    with open(PARAM_FILE, "rb") as f:
        params = pickle.load(f)
except Exception:
    safe_exit(1e9, "param_load_error")

dat_type = params["dat_type"]
img_size = params["img_size"]
training_data_path = params["training_data_path"]
validation_file_path = params["validation_file_path"]

input_shape = (img_size, img_size, 210)

# ============================================================
# Data loading
# ============================================================
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

# ============================================================
# Generator with cultivar
# ============================================================
def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_files = file_list[offset: offset + batch_size]
            batch_data, batch_targets, batch_cult = [], [], []

            for i, file in enumerate(batch_files):
                try:
                    data = np.load(file)
                    if img_size in (40, 20):
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

            expanded_cult = np.repeat(batch_cult[:, None, None, :], img_size, axis=1)
            expanded_cult = np.repeat(expanded_cult, img_size, axis=2)

            combined = np.concatenate([batch_data, expanded_cult], axis=-1)
            yield combined, batch_targets

# ============================================================
# ViT model
# ============================================================
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Add, Embedding, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def parse_tuple_string(x):
    """
    Convert strings like '128-64-32' into tuples (128, 64, 32).
    Leave everything else unchanged.
    """
    if isinstance(x, str) and "-" in x:
        parts = x.split("-")
        # Only convert if all parts are integers
        if all(p.isdigit() for p in parts):
            return tuple(int(p) for p in parts)
    return x

def create_vit_model(
    patch_size,
    projection_dim,
    transformer_layers,
    num_heads,
    mlp_head_units,
    dropout_rate,
):
    global input_shape

    mlp_head_units = list(mlp_head_units)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    inputs = tf.keras.Input(shape=input_shape)

    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="valid",
    )(inputs)
    patches = Reshape((-1, projection_dim))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + tf.expand_dims(position_embedding, axis=0)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=dropout_rate)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    outputs = Dense(1, activation="linear")(features)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=1e-3, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model

# ============================================================
# Training
# ============================================================
try:
    X_train, y_train, X_val, y_val, train_cult, val_cult = load_data(dat_type)

    train_gen = data_generator_w_cultivar(X_train, y_train, train_cult, batch_size, img_size)
    val_gen = data_generator_w_cultivar(X_val, y_val, val_cult, batch_size, img_size)

    if TEST_MODE:
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        train_cult = train_cult[:max_samples]

        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]
        val_cult = val_cult[:max_samples]

    model_keys = [
        "patch_size",
        "projection_dim",
        "transformer_layers",
        "num_heads",
        "mlp_head_units",
        "dropout_rate",
    ]

    model_params = {k: params[k] for k in model_keys}

    # Convert string mlp_head_units to tuple
    if "mlp_head_units" in model_params:
        model_params["mlp_head_units"] = parse_tuple_string(model_params["mlp_head_units"])

    model = create_vit_model(**model_params)

    early_stopping = EarlyStopping(
        monitor="val_mae",
        min_delta=0.001,
        patience=15,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=0,
        mode="min",
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=val_gen,
        validation_steps=len(X_val) // batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    try:
        val_loss = float(np.mean(history.history["val_loss"]))
        if np.isnan(val_loss) or np.isinf(val_loss):
            safe_exit(1e9, status="nan_or_inf")
    except Exception:
        safe_exit(1e9, status = "val_loss_error")

    safe_exit(val_loss, status = "ok")

except tf.errors.ResourceExhaustedError:
    safe_exit(1e9, status="oom")

except Exception:
    traceback.print_exc()
    safe_exit(1e9, status="crash")

finally:
    gc.collect()
    clear_session()