#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, date

start_time = datetime.now()
today = date.today().strftime('%Y-%m-%d')
img_size = 50
Code_run_ID = today + f'run_10Hybrid_{img_size}px'


# In[ ]:


save_file_path = '/media/2tbdisk3/data/Haidee/Results/'


# In[ ]:


# import os

# os.makedirs(Code_run_ID, exist_ok=True)

# print(f"Directory '{Code_run_ID}' created successfully!")


# In[ ]:


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
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


# Data generators help with managing memory when using large amounts of data

def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    missing_files = [] # List of missing files

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
                    batch_data.append(data)
                    batch_targets.append(targets[offset + i])
                    batch_cultivars.append(cultivars[offset + i])
                except FileNotFoundError:
                    missing_files.append(file)
                    print(f"File not found: {file}. Skipping...")
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

        # After the loop, print and save the missing files
        if missing_files:
            print(f"Missing files: {missing_files}")
            missing_files = []  # Clear the list after saving


# In[ ]:


# Assuming input shape is (20, 20, 210)
input_shape = (img_size, img_size, 210) # 204 spectral bands + 6 cultivar onehot encoding


# In[ ]:


# Define MLP block for Transformer
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


# Define CNN-Transformer Hybrid model
def create_hybrid_model(
    input_shape,
    cnn_filters=[8, 16, 32],  # Filters for CNN layers
    patch_size=2,  # Size of patches for Transformer
    projection_dim=64,  # Embedding dimension for Transformer
    transformer_layers=3,  # Number of Transformer blocks
    num_heads=4,  # Number of attention heads
    mlp_head_units=[128, 64],     # Hidden units in the final MLP head
    dropout_rate=0.2,  # Dropout rate
):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Feature Extraction Path
    # First convolutional block
    x_cnn = Conv2D(cnn_filters[0], kernel_size=3, padding='same', activation='relu')(inputs)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D(pool_size=2, strides=1, padding='same')(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)
    
    # Second convolutional block
    x_cnn = Conv2D(cnn_filters[1], kernel_size=3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D(pool_size=2, padding='same')(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)
    
    # Third convolutional block
    x_cnn = Conv2D(cnn_filters[2], kernel_size=3, padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = AveragePooling2D(pool_size=2, padding='same')(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)
    
    # Get CNN feature map dimensions
    cnn_shape = K.int_shape(x_cnn)
    
    # Transformer Path
    # Create patches using Conv2D
    patches = Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    
    # Reshape patches to sequence format
    patch_dims = patches.shape[1] * patches.shape[2]
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
    
    # Process Transformer output
    transformer_output = LayerNormalization(epsilon=1e-6)(encoded_patches)
    transformer_output = GlobalAveragePooling1D()(transformer_output)
    
    # Process CNN output
    cnn_output = Flatten()(x_cnn)
    
    # Combine CNN and Transformer features
    combined_features = Concatenate()([cnn_output, transformer_output])
    
    # MLP head for final prediction
    combined_features = BatchNormalization()(combined_features)
    combined_features = mlp(combined_features, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
    
    # Output layer for regression
    outputs = Dense(1, activation="linear")(combined_features)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model


# In[ ]:


checkpoint_brix = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_brix_hybrid.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)

# 添加早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    min_delta=0.001,  # 最小变化阈值
                    patience=15,      # 如果验证MAE在15轮内没有改善，则停止训练
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)  # 恢复最佳权重


# In[ ]:

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Create Hybrid CNN-Transformer model
    model_hybrid_brix = create_hybrid_model(
        input_shape=input_shape,
        cnn_filters=[64, 128, 256],
        patch_size=5,  # 使用更小的patch尺寸，增加patch数量
        projection_dim=64,
        transformer_layers=4,
        num_heads=4,
        mlp_head_units=[128, 64],
        dropout_rate=0.15,
    )
    
    # Compile the model
    model_hybrid_brix.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

model_hybrid_brix.summary()


# In[ ]:


# load all_data

if img_size == 30 or img_size == 20:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'


X_train_brix          = np.load(f'{training_data_path}X_train_all_years_Brix_shuffled.npy')
Y_train_brix          = np.load(f'{training_data_path}Y_train_all_years_Brix_shuffled.npy')
X_validate_brix       = np.load(f'{training_data_path}X_validate_all_years_Brix_shuffled.npy')
Y_validate_brix       = np.load(f'{training_data_path}Y_validate_all_years_Brix_shuffled.npy')
Frmness_encoder_shuffled  = np.load(f'{training_data_path}X_train_all_years_Brix_encoder_shuffled.npy')
validate_encoder        = np.load(f'{training_data_path}X_validate_all_years_Brix_encoder_shuffled.npy')


# In[ ]:

spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

X_train_brix = [spectral_path + file for file in X_train_brix]
X_validate_brix = [spectral_path + file for file in X_validate_brix]


# In[ ]:


batch_size = 16  # Set your batch size

csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}_history_brix_hybrid.csv", append=True)


# In[ ]:


train_generator = data_generator_w_cultivar(X_train_brix, Y_train_brix, Frmness_encoder_shuffled, batch_size, img_size=img_size)
val_generator = data_generator_w_cultivar(X_validate_brix, Y_validate_brix, validate_encoder, batch_size, img_size=img_size)


# In[ ]:


model_hybrid_brix.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=len(X_train_brix) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_validate_brix) // batch_size,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}_history_brix_hybrid.csv", append=True), 
        checkpoint_brix,
        early_stopping
    ],
)


# In[ ]:


model_hybrid_brix.save(f'{save_file_path}{Code_run_ID}_model_trained_hybrid_brix.keras')


# In[ ]:


del model_hybrid_brix
gc.collect()

brix_end = datetime.now()
print("总训练时间 Total training time:")
print(brix_end - start_time)
brix_total_time = brix_end - start_time
print(brix_total_time)
print(datetime.now()) 