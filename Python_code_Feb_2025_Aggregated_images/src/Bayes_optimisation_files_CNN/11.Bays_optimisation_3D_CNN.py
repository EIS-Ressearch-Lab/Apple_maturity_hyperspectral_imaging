from skopt import gp_minimize
from tqdm import tqdm  # Import tqdm for progress bar
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import traceback
import os 
# disable XLA - don't want it running inside Bayesian optimisation loop
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Select gpu to use


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
Conv3D,
MaxPooling3D,
Flatten, 
Dense,
BatchNormalization,
Dropout,
GlobalAveragePooling3D,
Input,
Reshape,
Concatenate,
Add,
Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.keras.losses import Huber   
import pandas as pd
import numpy as np
import gc
import pickle
import os
from datetime import datetime, date

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({
"layout_optimizer": False
})

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

print("Num GPUs Available: ", len(gpus))



import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

start_time = datetime.now()
testing = False # Set to True for testing the code quickly
today = date.today().strftime('%Y-%m-%d')
Code_run_ID = today + 'run_29_3D_CNN_bays_opt_40px'

np.random.seed(779)
tf.random.set_seed(779)

img_size = 40
spectral_bands = 204
spatial_size = img_size
input_shape_3d = (spectral_bands, spatial_size, spatial_size, 1)
input_shape_cultivar = (6,) 
batch_size = 8 

# path of optimisation data files
training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/'
# Using the same validation data from training data
validation_file_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'
save_file_path = '/media/2tbdisk3/data/Haidee/Results/'

if testing == True:
    data_type = ['brix']
else:
    data_type = ['brix', 'firmness', 'starch']
# Only run for starch
# data_type = ['starch']

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

def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size, spectral_bands = 204, augment=False):
    num_samples = len(file_list)
    missing_files = []  # List of missing files

    while True:  # Infinite loop for generator
        indices = np.arange(num_samples)
        if augment:
            # 每个epoch打乱数据顺序
            np.random.shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            # Load the batch of data from file paths
            batch_indices = indices[offset: offset + batch_size]
            batch_files = [file_list[i] for i in batch_indices]
            batch_data = []
            batch_targets = []
            batch_cultivars = []

            # File loading and handling - ensures model runs if file not found
            for i, file in enumerate(batch_files):
                try:
                    data = np.load(file)
                    # print(data.shape)

                    # Make sure shape is correct
                    if data.shape[2] >= spectral_bands:
                        data = data[:, :, :spectral_bands]  # Keep only the first 204 spectral bands

                    if augment and np.random.rand() > 0.5:
                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=1)  # Flip hporizontally

                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=0)  # Flip vertically

                        # Add tiny Gaussian sound absoprtion
                        if np.random.random() > 0.7:   
                            noise = np.random.normal(0, 0.02, data.shape)
                            data += noise

                        if np.random.random() > 0.8:
                            k = np.random.randint(1, 4)  # Number of rotations
                            data = np.rot90(data, k, axes=(0, 1))  # Rotate in the spatial dimension

                    data = np.transpose(data, (2,0,1)) # becomes (spectral_bands, height, width) 


                    if img_size == 40 or img_size ==20:
                        data_reduced = data[:, 5:-5, 5:-5] # Remove 5 pixels from each edge
                        batch_data.append(data_reduced)
                    else:
                        batch_data.append(data)
                    batch_targets.append(targets[batch_indices[i]])
                    batch_cultivars.append(cultivars[batch_indices[i]])
                except FileNotFoundError:
                    missing_files.append(file)
                    print(f"File not found: {file}. Skipping...")
                    continue

            if len(batch_data) == 0:
                continue  # Skip if no data loaded
            
            # Convert lists to numpy arrays
            batch_data = np.array(batch_data)  # Shape: (batch_size, 20, 20, 204)
            batch_targets = np.array(batch_targets)  # Shape: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)

            # (batch_size, spectral_bands, height, width) -> (batch_size, spectral_bands, height, width, 1)
            batch_data = np.expand_dims(batch_data, axis=-1)

            # 品种信息不需要转换为3D形式，直接传递给模型 # Cultivar information does not need to be converted to 3D format, directly passed to the model
            # 返回数据和目标值 # Return data and target values
            yield [batch_data, batch_cultivars], batch_targets




# Define space of hyperparameters
num_layers = Integer(1, 5, name="num_layers")  # Number of convolutional layers
filters1 = Integer(16, 128, name="filters1")  # Filters for first Conv2D layer  
filters2 = Integer(32, 256, name="filters2")  # Filters for second Conv2D layer
filters3 = Integer(32, 512, name="filters3")
filters4 = Integer(32, 256, name="filters4")
filters5 = Integer(32, 128, name="filters5")
kernel_size = Integer(2, 5, name="kernel_size")  # Kernel size
kernel_size1 = Integer(2, 5, name="kernel_size1")  # Kernel size
dropout = Real(0.1, 0.5, name="dropout")  # Dropout rate
pool_size = Integer(1, 3, name="pool_size")  # Pooling size


search_space = [
        num_layers,
        filters1,
        filters2,
        filters3,
        filters4,
        filters5,
        kernel_size,
        kernel_size1,
        dropout,
        pool_size
    ]


default_params = [5, 32, 64, 128, 160, 192, 2, 2, 0.15, 1]

assert len(default_params)==len(search_space), 'Error: check shapes!'





# Create model


def create_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, kernel_size1, dropout, pool_size, input_shape=input_shape_3d, input_shape_cultivar=input_shape_cultivar):

    clear_session()  # Clear previous models from memory
    spectral_input = Input(shape=input_shape, name='spectral_input')

    #CNN blocks
    # Block1
    x = Conv3D(
        filters1,
        kernel_size=(kernel_size, kernel_size1, kernel_size1),
        padding = 'same')(spectral_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_Res = Conv3D(filters1, kernel_size=(kernel_size, kernel_size1, kernel_size1), padding='same')(x)
    x = Add()([x, x_Res])  # Residual connection
    x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
    x = Dropout(dropout)(x)

    #2nd layer
    x = Conv3D(
        filters2,
        kernel_size=(kernel_size, kernel_size1, kernel_size1), 
        padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
    x = Dropout(dropout)(x)

    if num_layers >= 3:
        x = Conv3D(
        filters3,
        kernel_size=(kernel_size, kernel_size1, kernel_size1), 
        padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # Add another convolutional layer to maintain the same spatial dimensions
        x = Conv3D(filters3, kernel_size=(kernel_size, kernel_size1, kernel_size1), padding='same')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
        x = Dropout(dropout)(x)

    if num_layers >= 4:
        x = Conv3D(
            filters4,
            kernel_size=(kernel_size, kernel_size1, kernel_size1),
            padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(filters4, kernel_size=(kernel_size, kernel_size1, kernel_size1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(x)
        x = Dropout(dropout)(x)

    if num_layers == 5:
        x = Conv3D(
            filters5,
            kernel_size=(kernel_size, kernel_size1, kernel_size1),
            padding = 'same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)


    x = GlobalAveragePooling3D()(x)

    cultivar_input = Input(shape=input_shape_cultivar, name='cultivar_input')
    merged = Concatenate()([x, cultivar_input])


    # Fully Connected Layers
    x = Dense(512)(merged)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output = Dense(1, name='output')(x)
    model = Model(inputs=[spectral_input, cultivar_input], outputs=output)


    return model




# Fitness function to optimise
best_loss = np.inf

@use_named_args(search_space)
def evaluate_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, kernel_size1, dropout, pool_size):

    global best_loss
    global path_best_model

    clear_session()
    gc.collect()

    print(f'num_layers: {num_layers}, filters1: {filters1}, filters2: {filters2}, filters3: {filters3}, filters4: {filters4}, filters5:{filters5}, kernel_size: {kernel_size}, kernel_size1: {kernel_size1}, dropout: {dropout}, pool_size: {pool_size}')

    # if not is_valid_configuration(num_layers, kernel_size):
    #     print(f"🔥 Invalid config due to spatial dimension collapse")
    #     return float('inf')  # Return a very high value to indicate that this configuration is invalid

    try:
        # # Load data
        # X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(data_type)
        if testing == True:
            n_epochs = 2
            train_generator = data_generator_w_cultivar(X_train[:8], y_train[:8], train_cultivars, batch_size=batch_size, img_size=img_size)
            val_generator = data_generator_w_cultivar(X_val[:8], y_val[:8], validate_encoder, batch_size=batch_size, img_size=img_size)
        else:
            n_epochs = 50
            train_generator = data_generator_w_cultivar(X_train, y_train, train_cultivars, batch_size=batch_size, img_size=img_size)
            val_generator = data_generator_w_cultivar(X_val, y_val, validate_encoder, batch_size=batch_size, img_size=img_size)

        # Create model
        model = create_model(num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, kernel_size1, dropout, pool_size)

        optimizer = Adam(
        learning_rate=0.0005,  # 将学习率从0.001降低到0.0005
        beta_1=0.9, 
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=1.0  # 梯度裁剪
    )

        huber_loss = Huber(delta=1.0)

        model.compile(
        optimizer=optimizer,
        loss=huber_loss,
        metrics=["mae"]
    )

        # Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=10)

        history = model.fit(train_generator, 
                            steps_per_epoch=len(X_train)//32, 
                            epochs=n_epochs, 
                            validation_data=val_generator, 
                            validation_steps=len(X_val)//32, 
                            verbose=0, 
                            callbacks=[early_stopping])


        # Evaluate model
        val_loss = np.mean(history.history["val_loss"])
        results_list.append([num_layers, filters1, filters2, filters3, filters4, filters5, kernel_size, kernel_size1, dropout, pool_size,  val_loss])


        if val_loss < best_loss:
            best_loss = val_loss
            model.save(path_best_model)

        del model
        gc.collect()
        clear_session()
        return val_loss

    except Exception as e:
        print(f"❌ An error occurred during optimization for configuration {num_layers, kernel_size, kernel_size1}: {e}")
        return 1e9  # Return a high value in case of error to skip this configuration



# Bayesian optimisation loop
for dat_type in data_type:

    def fitness_with_progress(params, data_type=dat_type):
        try:
           
            result = evaluate_model(params)
        except Exception as e:
            print(f"❌ An error occurred during evaluation for {dat_type}: {e}")
            traceback.print_exc()
            result = 1e9  # Return a high value in case of error to skip this configuration

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
            print(f"❌ An error occurred while saving checkpoint for {dat_type}: {e}")
        pbar.update(1)  # Update progress bar
        return result

    clear_session()
    gc.collect()
    print(f'Running optimisation for {dat_type}')
    path_best_model = f'/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/Bayes_opt_2026/{Code_run_ID}_3D_CNN_best_model_params_{dat_type}.keras'

    X_train, y_train, X_val, y_val, train_cultivars, validate_encoder = load_data(dat_type)

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}_{Code_run_ID}_model_file_{dat_type}_3D_CNN.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)

    # Set to 100 for real runs and 2 for testing
    if testing == True:
        n_calls = 12 
        # n_initial_points = 1 
    else:
        n_calls = 100 
        # n_initial_points = 10
    # Load checkpoint if exists
    checkpoint_path = f"{training_data_path}Bayes_opt_2026/{Code_run_ID}_3D_CNN_bayesian_optimization_checkpoint_{dat_type}.pkl"

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
                'num_layers', 'filters1', 'filters2', 'filters3', 'filters4', 'filters5',
                'kernel_size', 'kernel_size1', 'dropout', 'pool_size', 'val_loss'
            ])
            df_results.to_excel(f"{training_data_path}{Code_run_ID}bayesian_optimization_results_{dat_type}.xlsx", index=False)
            df_results.to_pickle(f"{training_data_path}{Code_run_ID}bayesian_optimization_results_{dat_type}.pkl")

            # Delete checkpoint file since run is complete
            os.remove(checkpoint_path)

            print(f'✅ Best Validation Loss for {dat_type}: {search_result.fun}')
            print(f'🏆 Best Parameters for {dat_type}:')
            print(f'   num_layers={search_result.x[0]}, filters1={search_result.x[1]}, filters2={search_result.x[2]},')
            print(f'   filters3={search_result.x[3]}, filters4={search_result.x[4]},filters5={search_result.x[5]}')
            print(f'   kernel_size={search_result.x[6]}, kernel_size1={search_result.x[7]}, dropout={search_result.x[8]}, pool_size={search_result.x[9]}')

        except Exception as e:
            print(f"❌ An error occurred during optimization for {dat_type}: {e}")
            continue

end_time = datetime.now()
print(end_time - start_time)