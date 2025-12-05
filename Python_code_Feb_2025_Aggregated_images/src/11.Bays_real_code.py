from skopt import gp_minimize
from tqdm import tqdm  # Import tqdm for progress bar
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import pandas as pd
import numpy as np
import gc
import pickle
import os

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

img_size = 40
input_shape = (img_size, img_size, 210)  # Load the data

# path of optimisation data files
training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/'
# Using the same validation data from training data
validation_file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

    



# data_type = ['brix', 'firmness', 'starch']
# Only run for starch
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
    


# Define space of hyperparameters
# search_space = [
#     Integer(1, 5, name="num_layers"),  # Number of convolutional layers
#     Integer(32, 124, name="filters1"),  # Filters for first Conv2D layer
#     Integer(32, 512, name="filters2"),  # Filters for second Conv2D layer
#     Integer(32, 1024, name="filters3"),
#     Integer(32, 2048, name="filters4"),
#     Integer(32, 1024, name="filters5"),
#     Integer(1, 2, name="kernel_size"),  # Kernel size
#     Real(0.1, 0.5, name="dropout")  # Dropout rate
#     # Real(1e-4, 1e-2, name="learning_rate")  # Learning rate
    
# ]

num_layers = Integer(1, 5, name="num_layers")  # Number of convolutional layers
filters1 = Integer(32, 124, name="filters1")  # Filters for first Conv2D layer  
filters2 = Integer(32, 512, name="filters2")  # Filters for second Conv2D layer
filters3 = Integer(32, 1024, name="filters3")
filters4 = Integer(32, 2048, name="filters4")
filters5 = Integer(32, 1024, name="filters5")
kernel_size = Integer(1, 2, name="kernel_size")  # Kernel size
dropout = Real(0.1, 0.5, name="dropout")  # Dropout rate
pool_size = Integer(1, 2, name="pool_size")  # Pooling size


search_space = [
    num_layers,
    filters1,
    filters2,
    filters3,
    filters4,
    filters5,
    kernel_size,
    dropout,
    pool_size
]

def constraint(params):
    num_layers = params[0]
    kernel_size = params[8]
    if kernel_size == 2 and num_layers > 3:
        return False
    return True


default_params = [3, 64, 128, 256, 512, 512, 1, 0.15, 1]

assert len(default_params)==len(search_space), 'Error: check shapes!'

# Create model

def create_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, dropout, pool_size):
    model = Sequential()
    
    # Convolutional Blocks
    model.add(Conv2D(filters1, kernel_size=(kernel_size, kernel_size), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding='same'))
    model.add(Dropout(dropout))

    #2nd layer
    model.add(Conv2D(filters2, kernel_size=(kernel_size, kernel_size), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), padding='same'))
    model.add(Dropout(dropout))

    if num_layers >= 3:
        model.add(tf.keras.layers.Conv2D(filters3, (kernel_size, kernel_size)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), padding='same'))
        model.add(tf.keras.layers.Dropout(dropout))
    if num_layers >= 4:
        model.add(tf.keras.layers.Conv2D(filters4, (kernel_size, kernel_size)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), padding='same'))
        model.add(tf.keras.layers.Dropout(dropout))
    if num_layers == 5:
        model.add(tf.keras.layers.Conv2D(filters5, (kernel_size, kernel_size)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), padding='same'))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(GlobalAveragePooling2D())
    
    # Fully Connected Layers
    model.add(Dense(1048, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
        
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["mae"])
    
    return model

def is_valid_configuration(num_layers, kernel_size):
    print(f"Simulating {num_layers} layers with kernel={kernel_size} ")
    # # Simulate spatial dimensions after num_layers of Conv2D
    # height, width = input_shape[0], input_shape[1]

    # for _ in range(num_layers):
    #     # simulate convolution with kernel_size and padding='valid'
    #     height = height - (kernel_size - 1)
    #     width = width - (kernel_size - 1)

    #     if height <= 0 or width <= 0:
    #         return False

    # return True
    height, width, _ = input_shape
    
    # Check if input size will collapse with number of layers and kernel size
    for layer in range(num_layers):
        height = (height - kernel_size + 1)  # Assuming 'same' padding
        width = (width - kernel_size + 1)
        
        # Ensure the image dimensions are not reduced to 1x1
        if height <= 1 or width <= 1:
            return False
    
    # If configuration is valid
    return True


# Fitness function to optimise
path_best_model = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/best_model.keras'
best_loss = np.inf

# @use_named_args(search_space)
def evaluate_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, dropout, pool_size, data_type):
    
    print(f"📊 Running model for data type: {data_type}")

    global best_loss
    global path_best_model


    print(f'num_layers: {num_layers}, filters1: {filters1}, filters2: {filters2}, filters3: {filters3}, filters4: {filters4}, filters5:{filters5}, kernel_size: {kernel_size}, dropout: {dropout}, pool_size: {pool_size}')

    if not is_valid_configuration(num_layers, kernel_size):
        print(f"🔥 Invalid config due to spatial dimension collapse")
        return float('inf')  # Return a very high value to indicate that this configuration is invalid
    
    try:
        # # Load data
        # X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(data_type)

        train_generator = data_generator_w_cultivar(X_train, y_train, train_cultivars, batch_size=32, img_size=img_size)
        val_generator = data_generator_w_cultivar(X_val, y_val, validate_encoder, batch_size=32, img_size=img_size)
    
        # Create model
        model = create_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, dropout, pool_size)

        # Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)

        history = model.fit(train_generator, 
                           steps_per_epoch=len(X_train)//32, 
                           epochs=50, 
                           validation_data=val_generator, 
                           validation_steps=len(X_val)//32, 
                           verbose=0, 
                           callbacks=[early_stopping])


        # Evaluate model
        val_loss = np.mean(history.history["val_loss"])
        results_list.append([num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, dropout, pool_size,  val_loss])


        if val_loss < best_loss:
            best_loss = val_loss
            model.save(path_best_model)

        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return val_loss
    
    except Exception as e:
        print(f"❌ An error occurred during optimization for configuration {num_layers, kernel_size}: {e}")
        return float('inf')  # Return a high value in case of error to skip this configuration

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
            def fitness_with_progress(params, data_type=type):
                result = evaluate_model(*params)
                tried_params.append(params)
                tried_scores.append(result)

                # Save checkpoint
                try:
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'x': tried_params,
                            'func_vals': tried_scores,
                            'results_list': results_list
                        }, f)
                except Exception as e:
                    print(f"❌ An error occurred while saving checkpoint for {data_type}: {e}")
                pbar.update(1)  # Update progress bar
                return result
            
               # Perform Bayesian optimization       
            search_result = gp_minimize(func=fitness_with_progress,   
                                dimensions=search_space,
                                acq_func='EI',    #  'gp_hedge'       
                                n_calls=n_calls,
                                random_state=779,
                                x0=tried_params if tried_params else default_params,
                                y0=tried_scores if tried_scores else None)

            # Save results to Excel
            df_results = pd.DataFrame(results_list, columns=[
                'num_layers', 'filters1', 'filters2', 'filters3', 'filters4', 'filters5',
                'kernel_size', 'dropout', 'pool_size', 'val_loss'
            ])
            df_results.to_excel(f"{training_data_path}bayesian_optimization_results_{type}.xlsx", index=False)
            df_results.to_pickle(f"{training_data_path}bayesian_optimization_results_{type}.pkl")

            # Delete checkpoint file since run is complete
            os.remove(checkpoint_path)

            print(f'✅ Best Validation Loss for {type}: {search_result.fun}')
            print(f'🏆 Best Parameters for {type}:')
            print(f'   num_layers={search_result.x[0]}, filters1={search_result.x[1]}, filters2={search_result.x[2]},')
            print(f'   filters3={search_result.x[3]}, filters4={search_result.x[4]},filters5={search_result.x[5]}')
            print(f'   kernel_size={search_result.x[6]}, dropout={search_result.x[7]}, pool_size={search_result.x[8]}')

        except Exception as e:
            print(f"❌ An error occurred during optimization for {type}: {e}")
            continue

 