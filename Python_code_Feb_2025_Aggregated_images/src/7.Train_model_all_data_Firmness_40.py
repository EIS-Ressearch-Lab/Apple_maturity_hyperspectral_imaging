#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import gc
from tensorflow.keras import backend as K

import datetime

import time
start_time = time.time()

today = datetime.date.today().strftime('%Y-%m-%d')
img_size = 40
Code_run_ID = today + f'run8_{img_size}px_bays_opt_parameters' 
batch_size = 32  # Set your batch size
input_shape = (img_size, img_size, 210) #250? - #wdith, height of bounding box, last 204 is number of spectral bands of the Specim IQ camera + 6 cultivar onehot encoding
# input_shape = (img_size, img_size, 204) # Without onehot encoding

# In[14]:


# save_file_path = '/media/2tbdisk1/data/Haidee/Training_results/Feb2025/all_years_results/'
save_file_path = '/media/2tbdisk3/data/Haidee/Results/'


# In[ ]:


# import os

# os.makedirs(Code_run_ID, exist_ok=True)

# print(f"Directory '{Code_run_ID}' created successfully!")


# In[41]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[42]:


from keras.backend import clear_session
import tensorflow

# Reset Keras Session # modified from https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/17 Jaycangel
def reset_keras():
    # Clear keras session
    clear_session()

    # clear global variables
    try:
        del classifier  # Update the variable name as needed
    except NameError:
        pass

    # force garbage collect
    gc.collect()

    # reset the GPU memory
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("Keras session reset and GPU memory cleared.")



# In[ ]:

# Clear the log file at the beginning if file exists

# with open(f"{Code_run_ID}/{today}_missing_files.log", "w") as log_file:
#     log_file.write("")  # Write an empty string to clear the file

# function to remove edge pixels from images

# Data generators help with managing memory when using large amounts of data

def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size, hotencoding = False):
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

            if hotencoding == True:
                batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)
            
                if len(batch_data) == 0:
                    continue  # Skip if no data loaded
            
                # Expand cultivar information to match the input data's spatial dimensions
                expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1)
                expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)
                # print(expanded_cultivars.shape)

                # Concatenate cultivar information with the original data along the last axis
                combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, 14, 14, 210)
            else:
                combined_data = batch_data

            # Yield the combined data and targets
            yield combined_data, batch_targets
        # break
 
        

    #     # After the loop, print and save the missing files
    #     if missing_files:
    #         print(f"Missing files: {missing_files}")
    #         # Append the missing files to a log file
    #         with open(f"/{Code_run_ID}/{today}_missing_files.log", "a") as log_file:
    #             for file in missing_files:
    #                 log_file.write(f"{file}\n")
    #         missing_files = []  # Clear the list after saving
    # num_samples = len(file_list)
    # missing_files = [] # List of missing files
        # break



# def data_generator(file_list, targets, batch_size):
#     num_samples = len(file_list)

#     while True: # Loop forever so the generator never terminates
#         for offset in range(0, num_samples, batch_size):
#             batch_files = file_list[offset : offset + batch_size]
#             batch_data = [np.load(file) for file in batch_files]

#             batch_targets = targets[offset : offset + batch_size]

#             # Convert lists to numpy arrays
#             batch_data = np.array(batch_data)
#             batch_targets = np.array(batch_targets)

#             yield batch_data, batch_targets



# In[36]:

# In[ ]:



# In[ ]:


checkpoint_firmness = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_firmness.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)



# In[ ]:
# Run 1
# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
#     # Create a Sequential model
#     model_complex_firmness = Sequential()

#     # Convolutional Block 1
#     model_complex_firmness.add(
#         Conv2D(64, kernel_size=(2, 2), activation="relu", input_shape=input_shape) # kernal size format is common choice for image data
#     )
#     model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2))) # cuts image in half
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 2
#     model_complex_firmness.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 3
#     model_complex_firmness.add(Conv2D(256, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 4
#     model_complex_firmness.add(Conv2D(512, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 5
#     model_complex_firmness.add(Conv2D(1024, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 6
#     # model_complex_firmness_only.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
#     # model_complex_firmness_only.add(BatchNormalization())
#     # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness_only.add(Dropout(0.25))

#     # Global Average Pooling
#     model_complex_firmness.add(GlobalAveragePooling2D())

#     # Dense layers
#     model_complex_firmness.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(1024, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(512, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(256, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     # Output layer with 1 neuron for regression
#     model_complex_firmness.add(Dense(3, activation="linear")) # This dense layer sets number of output nodes
    
#     # Compile the model
#     model_complex_firmness.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# model_complex_firmness.summary()


# # Run2&3& 4 
# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
#     # Create a Sequential model
#     model_complex_firmness = Sequential()

#     # Convolutional Block 1
#     model_complex_firmness.add(
#         Conv2D(64, kernel_size=(2, 2), activation="relu", input_shape=input_shape) # kernal size format is common choice for image data
#     )
#     model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2))) # cuts image in half
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 2
#     model_complex_firmness.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 3
#     model_complex_firmness.add(Conv2D(256, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 4
#     model_complex_firmness.add(Conv2D(512, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 5
#     model_complex_firmness.add(Conv2D(1024, kernel_size=(2, 2), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 6
#     # model_complex_firmness_only.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
#     # model_complex_firmness_only.add(BatchNormalization())
#     # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness_only.add(Dropout(0.25))

#     # Global Average Pooling
#     model_complex_firmness.add(GlobalAveragePooling2D())

#     # Dense layers
#     model_complex_firmness.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(1024, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(512, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(256, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     # Output layer with 1 neuron for regression
#     model_complex_firmness.add(Dense(1, activation="linear")) # This dense layer sets number of output nodes
    
#     # Compile the model
#     model_complex_firmness.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# model_complex_firmness.summary()


# # # Run 7 - bayesian optimization parameters - Side A 
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     # Create a Sequential model
#     model_complex_firmness = Sequential()

#     # Convolutional Block 1
#     model_complex_firmness.add(Conv2D(124, kernel_size=(1, 1), activation="relu", input_shape=input_shape))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2))) # cuts image in half
#     model_complex_firmness.add(Dropout(0.149302))

#     # Convolutional Block 2
#     model_complex_firmness.add(Conv2D(115, kernel_size=(1, 1), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.149302))

#     # Convolutional Block 3
#     model_complex_firmness.add(Conv2D(611, kernel_size=(1, 1), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.149302))

#     # Convolutional Block 4
#     model_complex_firmness.add(Conv2D(1384, kernel_size=(1, 1), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.149302))


#     # Convolutional Block 5
#     model_complex_firmness.add(Conv2D(612, kernel_size=(1, 1), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.149302))

#     # # Convolutional Block 6
#     # model_complex_firmness.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness.add(Dropout(0.25))

#     # Global Average Pooling
#     model_complex_firmness.add(GlobalAveragePooling2D())

#     # Dense layers
#     # model_complex_firmness.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(1024, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     # model_complex_firmness.add(Dense(512, activation="relu"))
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(256, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     # Output layer with 1 neuron for regression
#     model_complex_firmness.add(Dense(1, activation="linear")) # This dense layer sets number of output nodes
    
#     # Compile the model
#     model_complex_firmness.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# model_complex_firmness.summary()




# Run7 - all sides - bays opt parameters
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create a Sequential model
    model_complex_firmness = Sequential()

    # Convolutional Block 1
    model_complex_firmness.add(
        Conv2D(45, kernel_size=(1, 1), activation="relu", input_shape=input_shape))
    model_complex_firmness.add(BatchNormalization())
    model_complex_firmness.add(MaxPooling2D(pool_size=(1, 1), padding='same')) # cuts image in half
    model_complex_firmness.add(Dropout(0.157920))

    # # Convolutional Block 2
    # model_complex_firmness.add(Conv2D(128, kernel_size=(1, 1), activation="relu"))
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(MaxPooling2D(pool_size=(1, 1), padding = 'same'))
    # model_complex_firmness.add(Dropout(0.15))

    # # Convolutional Block 3
    # model_complex_firmness.add(Conv2D(256, kernel_size=(2, 2), activation="relu"))
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    # model_complex_firmness.add(Dropout(0.15))

    # # Convolutional Block 4
    # model_complex_firmness.add(Conv2D(512, kernel_size=(4, 4), activation="relu"))
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(MaxPooling2D(pool_size=(4, 4), padding = 'same'))
    # model_complex_firmness.add(Dropout(0.15))

    # # Convolutional Block 5
    # model_complex_firmness.add(Conv2D(1024, kernel_size=(2, 2), activation="relu"))
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    # model_complex_firmness.add(Dropout(0.25))

    # Convolutional Block 6
    # model_complex_firmness_only.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
    # model_complex_firmness_only.add(BatchNormalization())
    # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    # model_complex_firmness_only.add(Dropout(0.25))

    # Global Average Pooling
    model_complex_firmness.add(GlobalAveragePooling2D())

    # Dense layers
    # model_complex_firmness.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(Dropout(0.5))

    # model_complex_firmness.add(Dense(1024, activation="relu"))
    # model_complex_firmness.add(BatchNormalization())
    # model_complex_firmness.add(Dropout(0.5))

    model_complex_firmness.add(Dense(1048, activation="relu"))
    model_complex_firmness.add(BatchNormalization())
    model_complex_firmness.add(Dropout(0.5))

    model_complex_firmness.add(Dense(256, activation="relu"))
    model_complex_firmness.add(BatchNormalization())
    model_complex_firmness.add(Dropout(0.5))

    # Output layer with 1 neuron for regression
    model_complex_firmness.add(Dense(1, activation="linear")) # This dense layer sets number of output nodes
    
    # Compile the model
    model_complex_firmness.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

model_complex_firmness.summary()


# Run 10 baysian opt parameters
# strategy = tf.distribute.MirroredStrategy()

# with strategy.scope():
#     # Create a Sequential model
#     model_complex_firmness = Sequential()

#     # Convolutional Block 1
#     model_complex_firmness.add(
#         Conv2D(39, kernel_size=(1, 1), activation="relu", input_shape=input_shape) # kernal size format is common choice for image data
#     )
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(1, 1))) 
#     model_complex_firmness.add(Dropout(0.469233))

#     # Convolutional Block 2
#     model_complex_firmness.add(Conv2D(175, kernel_size=(1, 1), activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(MaxPooling2D(pool_size=(1, 1), padding = 'same'))
#     model_complex_firmness.add(Dropout(0.469233))

#     # # Convolutional Block 3
#     # model_complex_firmness.add(Conv2D(256, kernel_size=(1, 1), activation="relu"))
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(MaxPooling2D(pool_size=(1, 1), padding = 'same'))
#     # model_complex_firmness.add(Dropout(0.3994037022159196))

#     # # Convolutional Block 4
#     # model_complex_firmness.add(Conv2D(512, kernel_size=(2, 2), activation="relu"))
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness.add(Dropout(0.25))

#     # # Convolutional Block 5
#     # model_complex_firmness.add(Conv2D(1024, kernel_size=(2, 2), activation="relu"))
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness.add(Dropout(0.25))

#     # Convolutional Block 6
#     # model_complex_firmness_only.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
#     # model_complex_firmness_only.add(BatchNormalization())
#     # model_complex_firmness_only.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
#     # model_complex_firmness_only.add(Dropout(0.25))

#     # Global Average Pooling
#     model_complex_firmness.add(GlobalAveragePooling2D())

#     # Dense layers
#     # model_complex_firmness.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(Dropout(0.5))

#     # model_complex_firmness.add(Dense(1024, activation="relu"))
#     # model_complex_firmness.add(BatchNormalization())
#     # model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(258, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     model_complex_firmness.add(Dense(128, activation="relu"))
#     model_complex_firmness.add(BatchNormalization())
#     model_complex_firmness.add(Dropout(0.5))

#     # Output layer with 1 neuron for regression
#     model_complex_firmness.add(Dense(1, activation="linear")) # This dense layer sets number of output nodes
    
#     # Compile the model
#     model_complex_firmness.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error", metrics=["mae"]

# )

# model_complex_firmness.summary()

# In[36]:


# load all_data

if img_size == 30 or img_size == 20:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

X_train_Firmness          = np.load(f'{training_data_path}X_train_all_years_Firmness_shuffled.npy')
Y_train_Firmness          = np.load(f'{training_data_path}Y_train_all_years_Firmness_shuffled.npy')
X_validate_Firmness       = np.load(f'{training_data_path}X_validate_all_years_Firmness_shuffled.npy')
Y_validate_Firmness       = np.load(f'{training_data_path}Y_validate_all_years_Firmness_shuffled.npy')
Frmness_encoder_shuffled  = np.load(f'{training_data_path}X_train_all_years_Firmness_encoder_shuffled.npy')
validate_encoder        = np.load(f'{training_data_path}X_validate_all_years_Firmness_encoder_shuffled.npy')



# # Side A
# if img_size == 30 or img_size == 20:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/'
# elif img_size == 50 or img_size ==40:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/'

# X_train_Firmness          = np.load(f'{training_data_path}Side_A_X_train_all_years_Firmness_shuffled.npy')
# Y_train_Firmness          = np.load(f'{training_data_path}Side_A_Y_train_all_years_Firmness_shuffled.npy')
# X_validate_Firmness       = np.load(f'{training_data_path}Side_A_X_validate_all_years_Firmness_shuffled.npy')
# Y_validate_Firmness       = np.load(f'{training_data_path}Side_A_Y_validate_all_years_Firmness_shuffled.npy')
# Frmness_encoder_shuffled  = np.load(f'{training_data_path}Side_A_X_train_all_years_Firmness_encoder.npy')
# validate_encoder        = np.load(f'{training_data_path}Side_A_X_validate_all_years_Firmness_encoder.npy')
# In[40]:

spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

X_train_Firmness = [spectral_path + file for file in X_train_Firmness]
X_validate_Firmness = [spectral_path + file for file in X_validate_Firmness]





# In[ ]:



csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}history_firmness.csv", append=True)


# In[ ]:


train_generator = data_generator_w_cultivar(X_train_Firmness, Y_train_Firmness, Frmness_encoder_shuffled, batch_size, img_size, hotencoding=True)
val_generator = data_generator_w_cultivar(X_validate_Firmness, Y_validate_Firmness, validate_encoder, batch_size, img_size, hotencoding=True)


# In[ ]:


model_complex_firmness.fit(train_generator,epochs=110,steps_per_epoch=len(X_train_Firmness) // batch_size,validation_data=val_generator,validation_steps=len(X_validate_Firmness) // batch_size,callbacks=[tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}history_firmness.csv", append=True), checkpoint_firmness],)


# In[ ]:


# model_complex_firmness.save(f'{save_file_path}{Code_run_ID}_{today}model_trained_100epoch_firmness.keras')


# In[ ]:


del model_complex_firmness
gc.collect()

firmness_end = time.time()
print(firmness_end - start_time)
firmness_total_time = firmness_end - start_time
print(firmness_total_time)

