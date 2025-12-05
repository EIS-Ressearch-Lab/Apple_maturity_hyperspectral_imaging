from skopt import gp_minimize
import traceback
import pickle
from tqdm import tqdm  # Import tqdm for progress bar
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Embedding,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.backend import clear_session
import pandas as pd
import numpy as np
import gc



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import time

np.random.seed(779)
tf.random.set_seed(779)

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()


import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


start_time = time.time()

# Parse command line arguments

batch_size = 32
img_size = 40
input_shape = (img_size, img_size, 210)  # Load the data

# path of optimisation data files
training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/'
# Using the same validation data from training data
validation_file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'
save_file_path = '/media/2tbdisk3/data/Haidee/Results/'
    

# data_type = ['brix', 'firmness', 'starch']
data_type = ['starch']

def load_data(data_type):
    if data_type == 'brix':
        spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

        X_train = np.load(f'{training_data_path}X_train_brix_optimisation_May2025.npy')
        y_train = np.load(f'{training_data_path}Y_train_brix_optimisation_May2025.npy')                     
        X_val= np.load(f'{validation_file_path}X_validate_all_years_Brix_shuffled.npy')                 
        y_val= np.load(f'{validation_file_path}Y_validate_all_years_Brix_shuffled.npy')                
        train_cultivars = np.load(f'{training_data_path}X_encoder_brix_optimisation_May2025.npy')   
        validate_encoder = np.load(f'{validation_file_path}X_validate_all_years_Brix_encoder_shuffled.npy')
        
        #Test
        # X_train = X_train[:3]
        # y_train = y_train[:3]
        # X_val= X_val[:3]
        # y_val= y_val[:3]
        # train_cultivars = train_cultivars [:3]
        # validate_encoder= validate_encoder[:3]

        X_train = [spectral_path + file for file in X_train]
        X_val = [spectral_path + file for file in X_val]
    elif data_type == 'firmness':
        spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

        X_train = np.load(f'{training_data_path}X_train_firmness_optimisation_May2025.npy')
        y_train = np.load(f'{training_data_path}Y_train_firmness_optimisation_May2025.npy')                     
        X_val= np.load(f'{validation_file_path}X_validate_all_years_Firmness_shuffled.npy')                  
        y_val= np.load(f'{validation_file_path}Y_validate_all_years_Firmness_shuffled.npy')                
        train_cultivars = np.load(f'{training_data_path}X_encoder_firmness_optimisation_May2025.npy')   
        validate_encoder = np.load(f'{validation_file_path}X_validate_all_years_Firmness_encoder_shuffled.npy')
        X_train = [spectral_path + file for file in X_train]
        X_val = [spectral_path + file for file in X_val]
        
    elif data_type == 'starch':
        spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

        X_train = np.load(f'{training_data_path}X_train_starch_optimisation_May2025.npy')
        y_train = np.load(f'{training_data_path}Y_train_starch_optimisation_May2025.npy')                     
        X_val= np.load(f'{validation_file_path}X_validate_all_years_Starch_shuffled.npy')                  
        y_val= np.load(f'{validation_file_path}Y_validate_all_years_Starch_shuffled.npy')                
        train_cultivars = np.load(f'{training_data_path}X_encoder_starch_optimisation_May2025.npy')   
        validate_encoder = np.load(f'{validation_file_path}X_validate_all_years_Starch_encoder_shuffled.npy')
        X_train = [spectral_path + file for file in X_train]
        X_val = [spectral_path + file for file in X_val]
    return X_train, y_train, X_val, y_val, train_cultivars, validate_encoder



def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    missing_files = []  # List of missing files
    while True:  # Infinite loop for generator
        for offset in range(0, num_samples, batch_size):
            # Load the batch of data from file paths
            batch_files = file_list[offset: offset + batch_size]
            batch_data = []
            batch_targets = []
            batch_cultivars = []
            
            # File loading and handling - ensures model runs if file not found
            for i, file in enumerate(batch_files):
                try:
                    data = np.load(file)
                    # print(data.shape)
                    if img_size == 40 or img_size ==20:
                        data_reduced = data[5:-5, 5:-5, :] # Remove 5 pixels from each edge
                        batch_data.append(data_reduced)
                    else:
                        batch_data.append(data)
                    batch_targets.append(targets[offset + i])
                    batch_cultivars.append(cultivars[offset + i])
                except FileNotFoundError:
                    missing_files.append(file)
                    print(f"File not found: {file}. Skipping...")
                    continue
            
            # Convert lists to numpy arrays
            batch_data = np.array(batch_data)  # Shape: (batch_size, 20, 20, 204)
            # print(batch_data.shape)
            batch_targets = np.array(batch_targets)  # Shape: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)
            
            if len(batch_data) == 0:
                continue  # Skip if no data loaded
            
            # Expand cultivar information to match the input data's spatial dimensions
            expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1)
            expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)
            # print(expanded_cultivars.shape)

            # Concatenate cultivar information with the original data along the last axis
            combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, 14, 14, 210)
            
            # Yield the combined data and targets
            yield combined_data, batch_targets




search_space = [
    Integer(2, 10, name = "patch_size"),  # Adjust based on your image size
    Integer(64, 256, name = "projection_dim"),
    Integer(1, 5, name = "transformer_layers"),
    Integer(1, 5, name = "num_heads"),
    Categorical([
        "128-64",
        "256-128-64",
        "256-128"
    ], name="mlp_head_units"),
    Real(0.1, 0.5, name="dropout_rate")
]

reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)


# def constraint(params):
#     num_layers = params[0]
#     kernel_size = params[8]
#     if kernel_size == 2 and num_layers > 3:
#         return False
#     return True

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def parse_mlp_units(mlp_string):
    return [int(x) for x in mlp_string.split('-')]

def create_vit_model(
    input_shape,
    patch_size=5,  # Size of patches to extract from the input image
    projection_dim=128,  # Embedding dimension for each patch
    transformer_layers=4,  # Number of Transformer blocks
    num_heads=8,  # Number of attention heads
    mlp_head_units=list([256, 128, 64]),  # Hidden units in the MLP head
    dropout_rate=0.1,  # Dropout rate
):
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Create patches using Conv2D and Reshape
    patches = layers.Conv2D(
        filters=projection_dim,
        # kernel_size = patch_size,
        # strides = patch_size,
        kernel_size=(patch_size, patch_size),    #  tuple 
        strides=(patch_size, patch_size), # tuple
        padding="valid",
    )(inputs)
    patches = layers.Reshape((-1, projection_dim))(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    
    # Add position embeddings to patch embeddings
    # encoded_patches = patches + position_embedding
    encoded_patches = patches + tf.expand_dims(position_embedding, axis=0)
    
    # Create Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=dropout_rate
        )(x1, x1)
        
        # Skip connection 1
        x2 = Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=dropout_rate)
        
        # Skip connection 2
        encoded_patches = Add()([x3, x2])
    
    # Layer normalization
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Global average pooling
    representation = layers.GlobalAveragePooling1D()(representation)
    
    # MLP head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    
    # Output layer for regression
    outputs = Dense(1, activation="linear")(features)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# default_params = [3, 64, 128, 256, 512, 512, 1, 0.15, 1]

# assert len(default_params)==len(search_space), 'Error: check shapes!'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}_model_file_{type}_vit.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)


# Create model
def evaluate_vit_model(
    patch_size, projection_dim, transformer_layers, num_heads, mlp_head_units, dropout_rate,
    train_generator, val_generator, X_train, X_validate,
    batch_size, save_file_path, type,
    checkpoint, reduce_lr
):
    try:
        model = create_vit_model(
            input_shape=(img_size, img_size, 210),  # update based on your input
            patch_size=patch_size,
            projection_dim=projection_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            mlp_head_units=list(mlp_head_units),
            dropout_rate=dropout_rate
        )

        weight_decay_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: [
                w.assign(w * (1 - 1e-5)) 
                for w in model.trainable_weights 
                if 'kernel' in w.name
            ]
        )

        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

        early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    min_delta=0.001,  # 最小变化阈值
                    patience=15,      # 如果验证MAE在15轮内没有改善，则停止训练
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)  # 恢复最佳权重

       
        history = model.fit(
            train_generator,
            epochs=120,  # 增加训练轮次
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=val_generator,
            validation_steps=len(X_validate) // batch_size,
            callbacks=[
            tf.keras.callbacks.CSVLogger(f"{save_file_path}_history_{type}_vit.csv", append=True), 
            checkpoint,
            early_stopping,
            reduce_lr,
            weight_decay_callback
    ],
)

        val_loss = np.min(history.history["val_loss"])
        return val_loss

    except Exception as e:
        print(f"❌ Error: {e}")
        return float("inf")

# def is_valid_configuration(num_layers, kernel_size, input_shape):
#     print(f"Simulating {num_layers} layers with kernel={kernel_size} ")
#     # # Simulate spatial dimensions after num_layers of Conv2D
#     # height, width = input_shape[0], input_shape[1]

#     # for _ in range(num_layers):
#     #     # simulate convolution with kernel_size and padding='valid'
#     #     height = height - (kernel_size - 1)
#     #     width = width - (kernel_size - 1)

#     #     if height <= 0 or width <= 0:
#     #         return False

#     # return True
#     height, width, _ = input_shape
    
#     # Check if input size will collapse with number of layers and kernel size
#     for layer in range(num_layers):
#         height = (height - kernel_size + 1)  # Assuming 'same' padding
#         width = (width - kernel_size + 1)
        
#         # Ensure the image dimensions are not reduced to 1x1
#         if height <= 1 or width <= 1:
#             return False
    
#     # If configuration is valid
#     return True


# Fitness function to optimise
# path_best_model = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/best_model.keras'
best_loss = np.inf


# Bayesian optimisation loop
for type in data_type:
    
    clear_session()
    print(f'Running optimisation for {type}')
    X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(type)
    
    n_calls = 10

    # Load checkpoint if exists
    checkpoint_path = f"{training_data_path}bayesian_optimization_checkpoint_{type}.pkl"
    
    os.makedirs(training_data_path, exist_ok=True)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            tried_params = checkpoint['x']
            tried_scores = checkpoint['func_vals']
            results_list = checkpoint['results_list']
        print(f"🔁 Loaded checkpoint with {len(tried_params)} previous evaluations.")
    else:
        tried_params = []
        tried_scores = []
        results_list = []

    with tqdm(total=n_calls, desc="Optimisation Progress") as pbar:
        try: 
            # Define the fitness wrapper with a progress update
            # @use_named_args(search_space)
            def fitness_with_progress(params):
                # Map parameters to name
                param_names = ['patch_size', 'projection_dim', 'transformer_layers', 'num_heads', 'mlp_head_units', 'dropout_rate']
                param_dict = dict(zip(param_names, params))
                param_dict["mlp_head_units"] = parse_mlp_units(param_dict["mlp_head_units"])

                train_generator = data_generator_w_cultivar(X_train, y_train, train_cultivars, batch_size, img_size=img_size)
                val_generator = data_generator_w_cultivar(X_val, y_val, validate_encoder, batch_size, img_size=img_size)
    
                val_loss = evaluate_vit_model(
                    **param_dict,
                    train_generator=train_generator,
                    val_generator=val_generator,
                    X_train=X_train,
                    X_validate=X_val,
                    batch_size=batch_size,
                    save_file_path=save_file_path,
                    type=type,
                    checkpoint=checkpoint,
                    reduce_lr=reduce_lr
                )
                str_units = "-".join(map(str, param_dict["mlp_head_units"]))
                results_list.append([
                    param_dict['patch_size'],
                    param_dict['projection_dim'],
                    param_dict['transformer_layers'],
                    param_dict['num_heads'],
                    str_units,
                    param_dict['dropout_rate'],
                    val_loss
                ])

                tried_params.append(params)
                tried_scores.append(val_loss)

                
                # Save checkpoint
                try:
                   os.makedirs(training_data_path, exist_ok=True)
                   with open(checkpoint_path, 'wb') as f:
                       pickle.dump({
                           'x': tried_params,
                           'func_vals': tried_scores,
                           'results_list': results_list
                       }, f)
                   print(f"💾 Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                  print(f"❌ Failed to save checkpoint: {e}")


                pbar.update(1)  # Update progress bar
                return val_loss
            
               # Perform Bayesian optimization       
            search_result = gp_minimize(func=fitness_with_progress,   
                                dimensions=search_space,
                                acq_func='EI',    #  'gp_hedge'       
                                n_calls=n_calls,
                                random_state=779)

            # Save results to Excel
            df_results = pd.DataFrame(results_list, columns=[
                'patch_size',
                'projection_dim',
                'transformer_layers',
                'num_heads',
                'mlp_head_units',
                'dropout_rate',
                'val_loss'
            ])

            df_results.to_excel(f"{training_data_path}bayesian_optimization_results_{type}_ViT.xlsx", index=False)
            df_results.to_pickle(f"{training_data_path}bayesian_optimization_results_{type}_ViT.pkl")

            # Print best result summary
            print(f'✅ Best Validation Loss for {type}: {search_result.fun}')
            print(f'🏆 Best Parameters for {type}:')
            print(f'   patch_size = {search_result.x[0]}')
            print(f'   projection_dim = {search_result.x[1]}')
            print(f'   transformer_layers = {search_result.x[2]}')
            print(f'   num_heads = {search_result.x[3]}')
            print(f'   mlp_head_units = {search_result.x[4]}')
            print(f'   dropout_rate = {search_result.x[5]}')

        except Exception as e:
            print(f"❌ An error occurred during optimization for {type}: {e}")
            traceback.print_exc()
            continue


#Test code

# type = 'brix'
# clear_session()
# print(f'Running optimisation for {type}')
# X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(type)

# n_calls = 50
# results_list = []
# with tqdm(total=n_calls, desc="Optimisation Progress") as pbar:
#     try: 
#         # Define the fitness wrapper with a progress update
#         # @use_named_args(search_space)
#         def fitness_with_progress(params):
#             # Map parameters to name
#             param_names = ['patch_size', 'projection_dim', 'transformer_layers', 'num_heads', 'mlp_head_units', 'dropout_rate']
#             param_dict = dict(zip(param_names, params))
#             param_dict["mlp_head_units"] = parse_mlp_units(param_dict["mlp_head_units"])
#             train_generator = data_generator_w_cultivar(X_train, y_train, train_cultivars, batch_size, img_size=img_size)
#             val_generator = data_generator_w_cultivar(X_val, y_val, validate_encoder, batch_size, img_size=img_size)

#             val_loss = evaluate_vit_model(
#                 **param_dict,
#                 train_generator=train_generator,
#                 val_generator=val_generator,
#                 X_train=X_train,
#                 X_validate=X_val,
#                 batch_size=batch_size,
#                 save_file_path=save_file_path,
#                 type=type,
#                 checkpoint=checkpoint,
#                 reduce_lr=reduce_lr
#             )
#             str_units = "-".join(map(str, param_dict["mlp_head_units"]))
#             results_list.append([
#                 param_dict['patch_size'],
#                 param_dict['projection_dim'],
#                 param_dict['transformer_layers'],
#                 param_dict['num_heads'],
#                 str_units,
#                 param_dict['dropout_rate'],
#                 val_loss
#             ])
#             pbar.update(1)  # Update progress bar
#             return val_loss
        
#            # Perform Bayesian optimization       
#         search_result = gp_minimize(func=fitness_with_progress,   
#                             dimensions=search_space,
#                             acq_func='EI',    #  'gp_hedge'       
#                             n_calls=n_calls,
#                             random_state=779)
#         # Save results to Excel
#         df_results = pd.DataFrame(results_list, columns=[
#             'patch_size',
#             'projection_dim',
#             'transformer_layers',
#             'num_heads',
#             'mlp_head_units',
#             'dropout_rate',
#             'val_loss'
#         ])
#         df_results.to_excel(f"{training_data_path}bayesian_optimization_results_{type}_test.xlsx", index=False)
#         df_results.to_pickle(f"{training_data_path}bayesian_optimization_results_{type}_test.pkl")
#         # Print best result summary
#         print(f'✅ Best Validation Loss for {type}: {search_result.fun}')
#         print(f'🏆 Best Parameters for {type}:')
#         print(f'   patch_size = {search_result.x[0]}')
#         print(f'   projection_dim = {search_result.x[1]}')
#         print(f'   transformer_layers = {search_result.x[2]}')
#         print(f'   num_heads = {search_result.x[3]}')
#         print(f'   mlp_head_units = {search_result.x[4]}')
#         print(f'   dropout_rate = {search_result.x[5]}')
#     except Exception as e:
#         print(f"❌ An error occurred during optimization for {type}: {e}")
#         traceback.print_exc()
        
