#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import torch
import gc
from tensorflow.keras import backend as K

import datetime

import time
start_time = time.time()

today = datetime.date.today().strftime('%Y-%m-%d')
img_size = 50
Code_run_ID = today + 'run0_everything_model' + str(img_size) + 'px - no_OneHotEncoding' 
batch_size = 32  # Set your batch size
one_hot_encoding = False

# In[2]:


# save_file_path = '/media/2tbdisk1/data/Haidee/Training_results/Feb2025/all_years_results/'
save_file_path = '/media/2tbdisk3/data/Haidee/Results/'



# In[4]:



# In[5]:


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

# In[6]:


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
                    if img_size == 14:
                        data_reduced = data[3:-3, 3:-3, :] # Remove 3 pixels from each edge
                        batch_data.append(data_reduced)
                    elif img_size == 40:
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

#



# In[ ]:

if one_hot_encoding == True:
    input_shape = (img_size, img_size, 210) #250? - #wdith, height of bounding box, last 204 is number of spectral bands of the Specim IQ camera + 6 cultivar onehot encoding
else:
    input_shape = (img_size, img_size, 204)


# In[ ]:




checkpoint_var3 = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_var3.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)                    


# In[ ]:


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create a Sequential model
    model_complex_var3 = Sequential()

    # Convolutional Block 1
    model_complex_var3.add(
        Conv2D(64, kernel_size=(2, 2), activation="relu", input_shape=input_shape) # kernal size format is common choice for image data
    )
    model_complex_var3.add(BatchNormalization())
    # model_complex_var3_only.add(MaxPooling2D(pool_size=(2, 2))) # cuts image in half
    model_complex_var3.add(Dropout(0.25))

    # Convolutional Block 2
    model_complex_var3.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model_complex_var3.add(Dropout(0.25))

    # Convolutional Block 3
    model_complex_var3.add(Conv2D(256, kernel_size=(2, 2), activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model_complex_var3.add(Dropout(0.25))

    # Convolutional Block 4
    model_complex_var3.add(Conv2D(512, kernel_size=(2, 2), activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model_complex_var3.add(Dropout(0.25))

    # Convolutional Block 5
    model_complex_var3.add(Conv2D(1024, kernel_size=(2, 2), activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model_complex_var3.add(Dropout(0.25))

    # Convolutional Block 6
    # model_complex_var3_only.add(Conv2D(2048, kernel_size=(3, 3), activation="relu")) #2048 - number of filters
    # model_complex_var3_only.add(BatchNormalization())
    # model_complex_var3_only.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    # model_complex_var3_only.add(Dropout(0.25))

    # Global Average Pooling
    model_complex_var3.add(GlobalAveragePooling2D())

    # Dense layers
    model_complex_var3.add(Dense(2048, activation="relu")) # Dense layers other type of extraction method
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(Dropout(0.5))

    model_complex_var3.add(Dense(1024, activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(Dropout(0.5))

    model_complex_var3.add(Dense(512, activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(Dropout(0.5))

    model_complex_var3.add(Dense(256, activation="relu"))
    model_complex_var3.add(BatchNormalization())
    model_complex_var3.add(Dropout(0.5))

    # Output layer with 1 neuron for regression
    model_complex_var3.add(Dense(3, activation="linear")) # This dense layer sets number of output nodes
    
    # Compile the model
    model_complex_var3.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

model_complex_var3.summary()


# In[5]:


# In[5]:


# load all_data

# training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_Feb2025/all_years/'
if img_size == 30 or img_size == 20:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

training_data_path = training_data_path + 'all_3_variables/'

# X_train_Starch          = np.load(f'{training_data_path}X_train_all_years_Starch_shuffled.npy')
# Y_train_Starch          = np.load(f'{training_data_path}Y_train_all_years_Starch_shuffled.npy')
# X_validate_Starch       = np.load(f'{training_data_path}X_validate_all_years_Starch_shuffled.npy')
# Y_validate_Starch       = np.load(f'{training_data_path}Y_validate_all_years_Starch_shuffled.npy')
# Starch_encoder_shuffled  = np.load(f'{training_data_path}X_train_all_years_Starch_encoder_shuffled.npy')
# validate_encoder        = np.load(f'{training_data_path}X_validate_all_years_Starch_encoder_shuffled.npy')


X_train_var3          = np.load(f'{training_data_path}X_train_all_years_var3_shuffled.npy')
Y_train_var3          = np.load(f'{training_data_path}Y_train_all_years_var3_shuffled.npy')
X_validate_var3       = np.load(f'{training_data_path}X_validate_all_years_var3_shuffled.npy')
Y_validate_var3       = np.load(f'{training_data_path}Y_validate_all_years_var3_shuffled.npy')
var3_encoder_shuffled  = np.load(f'{training_data_path}X_train_all_years_var3_encoder_shuffled.npy')
validate_encoder        = np.load(f'{training_data_path}X_validate_all_years_var3_encoder_shuffled.npy')



# In[6]:



# In[8]:


spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

X_train_var3 = [spectral_path + file for file in X_train_var3]

X_validate_var3 = [spectral_path + file for file in X_validate_var3]




# In[14]:


# In[ ]:


batch_size = 32  # Set your batch size

csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}_history_var3.csv", append=True)


# In[ ]:


# train_generator = data_generator_w_cultivar(X_train_Starch, Y_train_Starch, Starch_encoder_shuffled, batch_size, img_size=img_size)
# val_generator = data_generator_w_cultivar(X_validate_Starch, Y_validate_Starch, validate_encoder, batch_size, img_size=img_size)
#all 3 variables
train_generator = data_generator_w_cultivar(X_train_var3, Y_train_var3, var3_encoder_shuffled, batch_size, img_size=img_size)#hotencoding=one_hot_encoding)
val_generator = data_generator_w_cultivar(X_validate_var3, Y_validate_var3, validate_encoder, batch_size, img_size=img_size) #hotencoding=one_hot_encoding)


# In[ ]:


model_complex_var3.fit(train_generator,epochs=110,steps_per_epoch=len(X_train_var3) // batch_size,validation_data=val_generator,validation_steps=len(X_validate_var3) // batch_size,callbacks=[tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}_history_var3_subset.csv", append=True), checkpoint_var3],)


# In[ ]:


# model_complex_var3.save(f'{save_file_path}{Code_run_ID}_{today}model_trained_100epoch_var3.keras')


# In[20]:


# del model_complex_var3
# gc.collect()

run_end = time.time()
print(run_end - start_time)
total_time = run_end - start_time
mins = total_time/60
print(f'total time {mins}minutes')


# In[ ]:




