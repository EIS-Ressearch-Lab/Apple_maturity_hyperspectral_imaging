#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU1


# In[1]:
print("running version 2")

from xml.parsers.expat import model
from shapley import ShapleyRegression
from stochastic_games import DatasetLossGame
import torch
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras.models import Sequential, Model

id = 'swapped_images'


gpus = tf.config.experimental.list_physical_devices('GPU')
print("Available GPUs:", gpus)

#enable memory growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)





# In[2]:


import shap
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sns



# In[3]:


img_size = 40
if img_size == 30 or img_size == 20:
    training_data_save_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_save_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'


# train data

X_train_all_years_Starch_shuffled = np.load(f'{training_data_save_path}X_train_all_years_Starch_shuffled.npy')
X_train_all_years_Brix_shuffled = np.load(f'{training_data_save_path}X_train_all_years_Brix_shuffled.npy')
X_train_all_years_Firmness_shuffled = np.load(f'{training_data_save_path}X_train_all_years_Firmness_shuffled.npy')


X_train_all_years_Starch_encoder   = np.load(f'{training_data_save_path}X_train_all_years_Starch_encoder_shuffled.npy')
X_train_all_years_Firmness_encoder = np.load(f'{training_data_save_path}X_train_all_years_Firmness_encoder_shuffled.npy')
X_train_all_years_Brix_encoder     = np.load(f'{training_data_save_path}X_train_all_years_Brix_encoder_shuffled.npy')

Y_train_Brix       = np.load(f'{training_data_save_path}Y_train_all_years_Brix_shuffled.npy')
Y_train_Starch       = np.load(f'{training_data_save_path}Y_train_all_years_Firmness_shuffled.npy')
Y_train_Firmness       = np.load(f'{training_data_save_path}Y_train_all_years_Starch_shuffled.npy')


# In[4]:


img_size = 40
batch_size = 24
num_batches = 5
File_path = '/media/2tbdisk3/data/Haidee/Results/'


# In[18]:


spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

# X_train is the same for all features
# X_train_all_years_Starch_shuffled = [spectral_path + file for file in X_train_all_years_Starch_shuffled] # use brix as the base
X_train_all_years_Brix_shuffled = [spectral_path + file for file in X_train_all_years_Brix_shuffled]
# X_train_all_years_Firmness_shuffled = [spectral_path + file for file in X_train_all_years_Firmness_shuffled]


# print(X_train_all_years_Starch_shuffled[:3])

X_train_all = X_train_all_years_Brix_shuffled
# In[ ]:



# In[20]:


import random
np.random.seed(771)
random.seed(771)

# Split the indices into cultivars
cultivars = ['Cox', 'Braeburn', 'Fuji', 'Gala', 'Golden Delicious', 'Jazz']
Cultivar_indices = {cultivar: [] for cultivar in cultivars}

for i, path in enumerate(X_train_all_years_Starch_shuffled):
    for cultivar in cultivars:
        if cultivar in path:
            Cultivar_indices[cultivar].append(i)
            

# Get the lengths of each group
group_lengths = {keyword: len(indices) for keyword, indices in Cultivar_indices.items()}
print(group_lengths)

# print(Cultivar_indices)

# Random split of indices into train and test sets by group
background_indices = {cultivar: [] for cultivar in cultivars}
for cultivar, indices in Cultivar_indices.items():
    random.shuffle(indices)
    split_index = int(0.05 * len(indices))  # 5% for background
    background_indices[cultivar] = indices[:split_index] 
    max_indices = int(0.6 *len(indices))
    Cultivar_indices[cultivar] = indices[split_index:max_indices]  # Use only 60% of the total data

# print(background_indices)

background_group_lengths = {keyword: len(indices) for keyword, indices in background_indices.items()}
print(background_group_lengths)

data_group_lengths = {keyword: len(indices) for keyword, indices in Cultivar_indices.items()}
print(data_group_lengths)

# Regroup the indices
# X_train = []
# X_train_encoder = []

X_train = [X_train_all_years_Brix_shuffled[i] for cultivar in cultivars for i in Cultivar_indices[cultivar]]
Y_train_brix = [Y_train_Brix[i] for cultivar in cultivars for i in Cultivar_indices[cultivar]]
Y_train_firmness = [Y_train_Firmness[i] for cultivar in cultivars for i in Cultivar_indices[cultivar]]
Y_train_starch = [Y_train_Starch[i] for cultivar in cultivars for i in Cultivar_indices[cultivar]]
X_train_encoder = [X_train_all_years_Brix_encoder[i] for cultivar in cultivars for i in Cultivar_indices[cultivar]]

X_background_sample = [X_train_all_years_Brix_shuffled[i] for cultivar in cultivars for i in background_indices[cultivar]]
Y_background_sample_brix = [Y_train_Brix[i] for cultivar in cultivars for i in background_indices[cultivar]]
Y_background_sample_firmness = [Y_train_Firmness[i] for cultivar in cultivars for i in background_indices[cultivar]]
Y_background_sample_starch = [Y_train_Starch[i] for cultivar in cultivars for i in background_indices[cultivar]]
X_background_encoder = [X_train_all_years_Brix_encoder[i] for cultivar in cultivars for i in background_indices[cultivar]]



print(len(X_train))
print(len(Y_train_brix))
print(len(X_train_encoder))
print(len(X_background_sample))
print(len(X_background_encoder))


Y_sample_brix = Y_train_brix
Y_sample_firmness = Y_train_firmness
Y_sample_starch = Y_train_starch

# In[ ]:


random.seed(771)
random.shuffle(X_train)
random.shuffle(Y_train_brix)
random.shuffle(Y_train_firmness)
random.shuffle(Y_train_starch)
random.shuffle(X_train_encoder)
random.shuffle(X_background_sample)
random.shuffle(X_background_encoder)


# In[8]:


# Load prediciton models
# prediction_model_brix = tf.keras.models.load_model(f"{File_path}2025-07-04run_20_40px_model_trained_vit_brix.keras")
# prediction_model_firmness = tf.keras.models.load_model(f"{File_path}2025-07-04run_20_40px_model_trained_vit_firmness.keras")
# prediction_model_starch = tf.keras.models.load_model(f"{File_path}2025-07-04run_20_40px_model_trained_vit_starch.keras")
##Swapped models
prediction_model_brix = tf.keras.models.load_model(f"{File_path}2025-07-10run_25_swapped_img_40px_model_trained_vit_brix.keras")
prediction_model_firmness = tf.keras.models.load_model(f"{File_path}2025-07-10run_25_swapped_img_40px_model_trained_vit_firmness.keras")
prediction_model_starch = tf.keras.models.load_model(f"{File_path}2025-07-10run_25_swapped_img_40px_model_trained_vit_starch.keras")

# In[9]:

# import pandas as pd

# path = "/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/"

# brix_dict = pd.read_pickle(f'{path}bayesian_optimization_results_brix_ViT.pkl')
# brix_min = brix_dict.copy()
# brix_min = brix_min.loc[brix_min['val_loss'].idxmin()]
# brix_min['mlp_head_units'] = list(map(int, str(brix_min['mlp_head_units']).split('-')))
# print(brix_min)
# print(brix_min['patch_size'])

# firmness_dict = pd.read_pickle(f'{path}bayesian_optimization_results_firmness_ViT.pkl')
# firmness_min = firmness_dict.copy()
# firmness_min = firmness_min.loc[firmness_min['val_loss'].idxmin()]
# firmness_min['mlp_head_units'] = list(map(int, str(firmness_min['mlp_head_units']).split('-')))
# print(firmness_min)

# starch_dict = pd.read_pickle(f'{path}bayesian_optimization_results_starch_ViT.pkl')
# starch_min = starch_dict.copy()
# starch_min = starch_min.loc[starch_min['val_loss'].idxmin()]
# starch_min['mlp_head_units'] = list(map(int, str(starch_min['mlp_head_units']).split('-')))
# print(starch_min)





def prep_data(file_list, cultivars, img_size):

    # Initialize lists to store the batch data
    batch_data = []
    batch_cultivars = []

    
    # Load the batch of data from file paths
    for i, file in enumerate(file_list):
        try:
            data = np.load(file)
            if img_size == 14:
                data_reduced = data[3:-3, 3:-3, :]
                batch_data.append(data_reduced)
            elif img_size == 40:
                data_reduced = data[5:-5, 5:-5, :]
                batch_data.append(data_reduced)
            else:
                batch_data.append(data)
            batch_cultivars.append(cultivars[i])
        except FileNotFoundError:
            print(f"File not found: {file}. Skipping...")
            continue

    # Convert lists to numpy arrays
    batch_data = np.array(batch_data)  # Shape: (batch_size, img_size, img_size, 204)
    batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)

    if len(batch_data) > 0:
        # Expand cultivar information to match the input data's spatial dimensions
        expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1)
        expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)

        # Concatenate cultivar information with the original data along the last axis
        combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, img_size, img_size, 210)

        return combined_data


# In[ ]:


# prediction_model_brix.summary()


# In[27]:



# sample_dat = prep_data(X_train, X_train_encoder, img_size, prediction_model_brix)
# background_dat = prep_data(X_background_sample, X_background_encoder, img_size, prediction_model_brix)
sample_dat = prep_data(X_train, X_train_encoder, img_size) # test subset
background_dat = prep_data(X_background_sample, X_background_encoder, img_size)
# print(sample_dat.shape)
# print(background_dat.shape)



# Wrap the model to only return the regression output
brix_model = tf.keras.Model(
    inputs=prediction_model_brix.input,
    outputs=prediction_model_brix.output["regression"]
)

firmness_model = tf.keras.Model(
    inputs=prediction_model_firmness.input,
    outputs=prediction_model_firmness.output["regression"]
)

starch_model = tf.keras.Model(
    inputs=prediction_model_starch.input,
    outputs=prediction_model_starch.output["regression"]
)

def run_shap_in_batches(model, background_dat, sample_dat, batch_size=16):
    """
    Computes SHAP values in batches to prevent GPU crashes.
    Returns the concatenated SHAP values across all batches.
    """
    explainer = shap.GradientExplainer(model, background_dat)
    shap_values_all = None

    for i in range(0, len(sample_dat), batch_size):
        batch = sample_dat[i:i + batch_size]
        try:
            shap_vals = explainer.shap_values(batch)

            # Initialize accumulation array if not set
            if shap_values_all is None:
                if isinstance(shap_vals, list):
                    shap_values_all = [np.array(s) for s in shap_vals]
                else:
                    shap_values_all = np.array(shap_vals)
            else:
                if isinstance(shap_vals, list):
                    for j in range(len(shap_vals)):
                        shap_values_all[j] = np.concatenate([shap_values_all[j], shap_vals[j]], axis=0)
                else:
                    shap_values_all = np.concatenate([shap_values_all, shap_vals], axis=0)

        except Exception as e:
            print(f"⚠️ SHAP batch {i}-{i + batch_size} failed: {e}")

    return shap_values_all

# In[ ]:
output_dir = f'/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Shap_values/May2025/{id}/'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created: {output_dir}")

data_types = ['brix', 'firmness', 'starch']
# data_types = ['firmness']
for data in data_types:
    print(f"\n=== Running SHAP analysis for {data}")

    model = globals()[f"{data}_model"]

    try:
        # explainer = shap.GradientExplainer(model, background_dat)
        # shap_values = explainer.shap_values(sample_dat)

        shap_values = run_shap_in_batches(model, background_dat, sample_dat, batch_size=16)

        # Save the SHAP values as pkl
        with open(f"{output_dir}shap_values_{data}_{id}.pkl", "wb") as f:
            pickle.dump(shap_values, f)
        print(f"SHAP values saved to '{output_dir}shap_values_{data}_{id}.pkl'.")

        # Save SHAP values as numpy array
        np.save(f"{output_dir}shap_values_{data}_{id}.npy", shap_values)
        print(f"SHAP values saved to '{output_dir}shap_values_{data}_{id}.npy'.")
    except Exception as e:
        print(f"❌ SHAP analysis failed for {data}: {e}")


    









