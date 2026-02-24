import pickle

# ---------------------------------------------------------
# Choose which model you want to test:
#   "2D_CNN", "3D_CNN", "hybrid", or "ViT"
# ---------------------------------------------------------
MODEL = "hybrid" 




default_params = {

    "2D_CNN": {
        "num_layers": 2,
        "filters1": 64,
        "filters2": 128,
        "filters3": 256,
        "filters4": 256,
        "filters5": 128,
        "kernel_size": 1,
        "dropout": 0.3,
        "pool_size": 1,
    },

    "3D_CNN": {
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
    },

    "hybrid": {
        "CNN_layers": 2,
        "filter1": 32,
        "filter2": 64,
        "filter3": 128,
        "kernel": 3,
        "dropout": 0.2,
        "pool_size": 2,
        "transformer_layers": 2,
        "patch_size": 4,
        "projection_dim": 64,
        "num_heads": 2,
        "mlp_head_units": "128-64", # or tuple (128, 64)
    },

    "ViT": {
        "patch_size": 4,
        "projection_dim": 128,
        "transformer_layers": 2,
        "num_heads": 2,
        "mlp_head_units": "128-64", # or tuple (128, 64),
        "dropout_rate": 0.2,
    }
}

# ---------------------------------------------------------
# Add required metadata (your workers expect these)
# ---------------------------------------------------------
params = default_params[MODEL]

params["dat_type"] = "brix"
params["img_size"] = 40
params["training_data_path"] = "/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/"
params["validation_file_path"] = "/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/"
params["gpu_id"] = 0   # force GPU 0 for testing

# ---------------------------------------------------------
# Save to file
# ---------------------------------------------------------
with open(f"{MODEL}_test_params.pkl", "wb") as f:
    pickle.dump(params, f)

print("Created test_params.pkl for model:", MODEL)


