#!/usr/bin/env python
# coding: utf-8

# In[25]:


from tkinter.filedialog import test
from skopt import gp_minimize
from tqdm import tqdm  # Import tqdm for progress bar
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import traceback
import os 
# disable XLA - don't want it running inside Bayesian optimisation loop
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Embedding,
    Reshape,
    Concatenate,
    AveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import gc
import pickle
from datetime import datetime, date
import time
from functools import partial




# In[2]:


tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": False
})
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()


# In[3]:


start_time = datetime.now()
testing = False # Set to True for testing the code quickly
today = date.today().strftime('%Y-%m-%d')
img_size = 40
Code_run_ID = today + 'run_29_hybrid_bays_opt_40px'

np.random.seed(779)
tf.random.set_seed(779)

input_shape = (img_size, img_size, 210)  # Load the data
batch_size = 32

# path of optimisation data files
training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/'
# Using the same validation data from training data
validation_file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'
save_file_path = '/media/2tbdisk3/data/Haidee/Results/'


# In[4]:


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
    


# In[27]:


def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    # missing_files = [] # List of missing files

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
                    # missing_files.append(file)
                    # print(f"File not found: {file}. Skipping...")
                    continue
            

            # Convert lists to numpy arrays
            batch_data = np.array(batch_data)  # Shape: (batch_size, 20, 20, 204)
            batch_targets = np.array(batch_targets)  # Shape: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)
            
            if len(batch_data) == 0:
                continue # Skip if no data loaded
            
                       
            # Expand cultivar information to match the input data's spatial dimensions
            expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1) # Adds singleton dimensions to match the input data's spatial dimensions (batchsize, 1, 1, 1, 6)
            expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)

            # Concatenate cultivar information with the original data along the last axis
            combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, 20, 20, 210)


            # Yield the combined data and targets
            yield combined_data, batch_targets

        # # After the loop, print and save the missing files
        # if missing_files:
        #     print(f"Missing files: {missing_files}")
        #     missing_files = []  # Clear the list after saving


# In[6]:


# Define MLP block for Transformer
def mlp(x, hidden_units, dropout):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout)(x)
    return x

def parse_string(x):
    if isinstance(x, str):
        return tuple(int(v) for v in x.split("-"))
    return x  # already a tuple


# In[7]:


# dummy search space to test optimisation loop
search_space = [
    Integer(1, 3, name="CNN_layers"),
    Integer(8, 64, name="filter1"),
    Integer(8, 128, name="filter2"),
    Integer(32, 256, name="filter3"),
    # Integer(32, 256, name="filter4"),
    Integer(2, 5, name="kernel"),
    Real(0.1, 0.4, name="dropout"),
    Integer(1, 3, name="pool_size"),
    Integer(1, 5, name="transformer_layers"),
    Integer(2, 10, name = "patch_size"),
    Categorical(
    [32, 64, 96, 128, 160, 192, 224, 256], name="projection_dim"),
    Categorical([1, 2, 4, 8], name="num_heads"),
    Categorical([
    (128, 64),
    (256, 128, 64),
    (256, 128)], name="mlp_head_units"),
    
]




# In[8]:


default_params = [
    3,          # CNN_layers
    8,          # filter1
    16,         # filter2
    32,         # filter3
    # 64,         # filter4
    3,          # kernel
    0.2,        # dropout
    2,          # pool_size
    3,          # transformer_layers
    2,          # patch_size
    64,         # projection_dim
    4,          # num_heads
    (128, 64),  # mlp_head_units
]
assert len(default_params)==len(search_space), 'Error: check shapes!'


# In[44]:


# Define CNN-Transformer hybrid model
def create_hybrid_model(
    CNN_layers=3,
    filter1=8,
    filter2=16,
    filter3=32,
    # filter4=64,
    kernel=3,
    dropout=0.2,  # Dropout rate
    pool_size=2,
    transformer_layers=3,  # Number of Transformer blocks
    patch_size=2,  # Size of patches for Transformer
    projection_dim=64,  # Embedding dimension for Transformer
    num_heads=4,  # Number of attention heads
    mlp_head_units=(128, 64)     # Hidden units in the final MLP head
    
    
):
    global input_shape  
    mlp_head_units = list(mlp_head_units) 

    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Feature Extraction Path
    # First convolutional block
    x_cnn = Conv2D(filter1, kernel_size=(kernel, kernel), padding='same', activation='relu')(inputs)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D(pool_size=(pool_size, pool_size), strides=1, padding='same')(x_cnn)
    x_cnn = Dropout(dropout)(x_cnn)
    
    if CNN_layers >= 2: # If fewer transformer layers, add more CNN layers to balance
        # Second convolutional block
        x_cnn = Conv2D(filter2, kernel_size=(kernel, kernel), padding='same', activation='relu')(x_cnn)
        x_cnn = BatchNormalization()(x_cnn)
        x_cnn = MaxPooling2D(pool_size=(pool_size, pool_size), padding='same')(x_cnn)
        x_cnn = Dropout(dropout)(x_cnn)
    
    # Third convolutional block
    if CNN_layers >= 3: # If even fewer transformer layers, add more CNN layers to balance
        x_cnn = Conv2D(filter3, kernel_size=(kernel, kernel), padding='same', activation='relu')(x_cnn)
        x_cnn = BatchNormalization()(x_cnn)
        x_cnn = AveragePooling2D(pool_size=(pool_size, pool_size), padding='same')(x_cnn)
        x_cnn = Dropout(dropout)(x_cnn)

    # if CNN_layers >= 4:
    #     x_cnn = Conv2D(filter4, kernel_size=(kernel, kernel), padding='same', activation='relu')(x_cnn)
    #     x_cnn = BatchNormalization()(x_cnn)
    #     x_cnn = AveragePooling2D(pool_size=(pool_size, pool_size), padding='same')(x_cnn)
    #     x_cnn = Dropout(dropout)(x_cnn)
    
    # Get CNN feature map dimensions
    cnn_shape = K.int_shape(x_cnn)
    
    # Transformer Path
    # Create patches using Conv2D
    patches = Conv2D(
        filters=projection_dim,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="valid",
    )(inputs)
    
    # Reshape patches to sequence format
    patch_dims = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patches = Reshape((-1, projection_dim))(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=patch_dims, delta=1)
    position_embedding = Embedding(
        input_dim=patch_dims, output_dim=projection_dim
    )(positions)
    
    # Add position embeddings to patch embeddings
    encoded_patches = patches + position_embedding
    
    # Create Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=dropout
        )(x1, x1)
        
        # Skip connection 1
        x2 = Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout=dropout)
        
        # Skip connection 2
        encoded_patches = Add()([x3, x2])
    
    # Process Transformer output
    transformer_output = LayerNormalization(epsilon=1e-6)(encoded_patches)
    transformer_output = GlobalAveragePooling1D()(transformer_output)
    
    # Process CNN output
    cnn_output = Flatten()(x_cnn)
    
    # Combine CNN and Transformer features
    combined_features = Concatenate()([cnn_output, transformer_output])
    
    # MLP head for final prediction
    combined_features = BatchNormalization()(combined_features)
    combined_features = mlp(combined_features, hidden_units=mlp_head_units, dropout=dropout)
    
    # Output layer for regression
    outputs = Dense(1, activation="linear")(combined_features)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# In[42]:


best_loss = float('inf')

def evaluate_hybrid_model(
    CNN_layers,
    filter1,
    filter2,
    filter3,
    # filter4
    kernel,
    dropout,
    pool_size,
    transformer_layers,
    patch_size,
    projection_dim,
    num_heads,
    mlp_head_units,
    dat_type):
    
    global best_loss
    global path_best_model

    K.clear_session()
    
    try:
        if testing == True:
            dat_type = 'brix'
            X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(dat_type)
            train_generator = data_generator_w_cultivar(X_train[:3], y_train[:3], train_cultivars[:3], batch_size, img_size=img_size)
            val_generator = data_generator_w_cultivar(X_val[:3], y_val[:3], validate_encoder[:3], batch_size, img_size=img_size)
            n_epochs = 2
        else:
   
            X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(dat_type)
            train_generator = data_generator_w_cultivar(X_train, y_train, train_cultivars, batch_size, img_size=img_size)
            val_generator = data_generator_w_cultivar(X_val, y_val, validate_encoder, batch_size, img_size=img_size)
            n_epochs = 50

        model = create_hybrid_model(
            CNN_layers=CNN_layers,
            filter1=filter1,
            filter2=filter2,
            filter3=filter3,
            kernel=kernel,
            dropout=dropout,
            pool_size=pool_size,
            patch_size=patch_size,
            projection_dim=projection_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            mlp_head_units=list(mlp_head_units)
        )
        
        optimizer = Adam(
            learning_rate=1e-3,
            weight_decay=1e-5
            )

        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

        early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    min_delta=0.001,  # 最小变化阈值
                    patience=15,      # 如果验证MAE在15轮内没有改善，则停止训练
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)  # 恢复最佳权重

        print("Model compiled successfully")
        print("Starting training...")
       
        history = model.fit(
            train_generator,
            epochs=n_epochs,  # 增加训练轮次
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=val_generator,
            validation_steps=len(X_val) // batch_size,
            callbacks=[
            early_stopping,
            model_checkpoint_cb
            ],)   

        try:
            val_loss = np.mean(history.history["val_loss"])
            if np.isnan(val_loss) or np.isinf(val_loss):
                print("Validation loss is NaN or Inf. Returning infinity for this configuration.")
                return float(1e9)
        except Exception:
            return float(1e9)

        results_list.append([
            CNN_layers, filter1, filter2, filter3, kernel, dropout, pool_size, transformer_layers, patch_size, projection_dim, num_heads, mlp_head_units, val_loss
            ])

        if val_loss < best_loss:
            best_loss = val_loss
            model.save(path_best_model)

        del model
        gc.collect()
        K.clear_session()

        return float(val_loss)

    except Exception as e:
        print(f"❌ Error: {e}")
        return float(1e9)


# In[ ]:


# if projection_dim % num_heads != 0:
#     return 1e9


# In[12]:


@use_named_args(search_space)
def fitness_with_progress(
    CNN_layers,
    filter1,
    filter2,
    filter3,
    # filter4
    kernel, 
    dropout,
    pool_size,
    transformer_layers,
    patch_size,
    projection_dim,
    num_heads,
    mlp_head_units):
                
                print('Testing:', 
                    'CNN_layer:', CNN_layers,
                    'filter1:', filter1,
                    'filter2:', filter2,
                    'filter3:', filter3,
                    # 'filter3:', filter4
                    'kernel:', kernel, 
                    'dropout:', dropout,
                    'pool_size:', pool_size,
                    'transformer_layers:', transformer_layers,
                    'patch_size:', patch_size,
                    'projection_dim:', projection_dim,
                    'num_heads:', num_heads,
                    'mlp_head_units:', mlp_head_units)
                # Map parameters to name
                # param_names = ['patch_size', 'projection_dim', 'transformer_layers', 'num_heads', 'mlp_head_units', 'dropout']
                # param_dict = dict(zip(param_names, params))
                # param_dict["mlp_head_units"] = parse_mlp_units(param_dict["mlp_head_units"])

                params = [CNN_layers, filter1, filter2, filter3, kernel, dropout, pool_size, transformer_layers, patch_size, projection_dim, num_heads, mlp_head_units]
                result = evaluate_hybrid_model(
                    CNN_layers, filter1, filter2, filter3, kernel, dropout, pool_size, transformer_layers, patch_size, projection_dim, num_heads, mlp_head_units, dat_type=dat_type
                )
                tried_params.append(params)
                tried_scores.append(result)
                # str_units = "-".join(map(str, param_dict["mlp_head_units"]))
                

                
                # Save checkpoint
                try:
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
                return float(result)


# In[34]:


print(search_space)


# In[13]:


data_type = ['brix', 'firmness', 'starch']


# In[45]:


for dat_type in data_type:
    
    K.clear_session()
    print(f'Running optimisation for {dat_type}')
    X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(dat_type)
    path_best_model = f'/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/Bayes_opt_2026/{Code_run_ID}_hybrid_best_model_params_{dat_type}.keras'

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}_{Code_run_ID}_model_file_{dat_type}_hybrid.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)

    
    if testing == True:
        n_calls = 12
    else:
        n_calls = 100

    # Load checkpoint if exists
    checkpoint_path = f"{training_data_path}Bayes_opt_2026/{Code_run_ID}hybrid_bayesian_optimization_checkpoint_{dat_type}.pkl"
    
    os.makedirs(training_data_path, exist_ok=True)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            tried_params = checkpoint['x']
            for i, params in enumerate(tried_params):
                if isinstance(params[11], str):
                    tried_params[i][11] = parse_string(params[11])
            tried_scores = checkpoint['func_vals']
            results_list = checkpoint['results_list']
            tried_params = [list(params) for params in tried_params]
        print(f"🔁 Loaded checkpoint with {len(tried_params)} previous evaluations.")

    else:
        tried_params = []
        tried_scores = []
        results_list = []

    with tqdm(total=n_calls, desc="Optimisation Progress") as pbar:
        try: 
            
            # Perform Bayesian optimization       
            search_result = gp_minimize(
                func=fitness_with_progress,   
                dimensions=search_space,
                acq_func='EI',    #  'gp_hedge'       
                n_calls=n_calls,
                random_state=779,
                x0=tried_params if tried_params else [default_params],
                y0=tried_scores if tried_scores else None)

            # Save results to Excel
            df_results = pd.DataFrame(results_list, columns=[
                'CNN_layers', 
                'filter1', 
                'filter2', 
                'filter3', 
                'kernel', 
                'dropout_rate', 
                'pool_size', 
                'transformer_layers', 
                'patch_size', 
                'projection_dim', 
                'num_heads', 
                'mlp_head_units',
                'val_loss'
            ])

            df_results.to_excel(f"{training_data_path}{Code_run_ID}_bayesian_optimization_results_{dat_type}_hybrid.xlsx", index=False)
            df_results.to_pickle(f"{training_data_path}{Code_run_ID}_bayesian_optimization_results_{dat_type}_hybrid.pkl")


            # Print best result summary
            print(f'🏆 Best Parameters for {dat_type}:')
            print(f'   CNN_layers        = {search_result.x[0]}')
            print(f'   filter1           = {search_result.x[1]}')
            print(f'   filter2           = {search_result.x[2]}')
            print(f'   filter3           = {search_result.x[3]}')
            print(f'   kernel            = {search_result.x[4]}')
            print(f'   dropout_rate      = {search_result.x[5]}')
            print(f'   pool_size         = {search_result.x[6]}')
            print(f'   transformer_layers= {search_result.x[7]}')
            print(f'   patch_size        = {search_result.x[8]}')
            print(f'   projection_dim    = {search_result.x[9]}')
            print(f'   num_heads         = {search_result.x[10]}')
            print(f'   mlp_head_units    = {search_result.x[11]}')


        except Exception as e:
            print(f"❌ An error occurred during optimization for {dat_type}: {e}")
            traceback.print_exc()
            continue


# In[ ]:
end_time = datetime.now()
print(end_time - start_time)

