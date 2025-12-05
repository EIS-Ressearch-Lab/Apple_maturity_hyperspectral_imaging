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
import pickle
from tqdm import tqdm

start_time = datetime.now()
today = date.today().strftime('%Y-%m-%d')
img_size = 40
band_filter = None
Code_run_ID = today + f'run_27_no_cultivar_{img_size}px'
channels = 204


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
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Embedding,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


# 尝试启用混合精度训练 - 兼容多个TensorFlow版本
try:
    # 较新版本的TensorFlow (2.4+)
    from tensorflow.keras.mixed_precision import global_policy, set_global_policy, Policy
    policy = Policy('mixed_float16')
    set_global_policy(policy)
    print('使用新版API设置混合精度')
except (ImportError, AttributeError):
    try:
        # 旧版本的TensorFlow (2.1-2.3)
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print('使用旧版API设置混合精度')
    except (ImportError, AttributeError):
        # 如果两种方法都失败，则不使用混合精度
        print('无法设置混合精度训练，将使用默认精度(float32)')


# In[ ]:
boolean_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Pickle_files/Shap_values/May2025/Filtered_booleans/'
with open(f'{boolean_path}booleans.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

if band_filter == 20:
    print("Using 20% of the features")
    selected_features_threshold_brix = loaded_data['selected_features_threshold_brix_20']
    selected_features_threshold_firmness = loaded_data['selected_features_threshold_firmness_20']
    selected_features_threshold_starch = loaded_data['selected_features_threshold_starch_20']
elif band_filter == 50:
    print("Using 50% of the features")
    selected_features_threshold_brix = loaded_data['selected_features_threshold_brix_50']
    selected_features_threshold_firmness = loaded_data['selected_features_threshold_firmness_50']
    selected_features_threshold_starch = loaded_data['selected_features_threshold_starch_50']
elif band_filter == 80:
    print("Using 80% of the features")
    selected_features_threshold_brix = loaded_data['selected_features_threshold_brix_80']
    selected_features_threshold_firmness = loaded_data['selected_features_threshold_firmness_80']
    selected_features_threshold_starch = loaded_data['selected_features_threshold_starch_80']


# Function specific to 40x40 images
def swap_quadrants_fast(image):
    swapped = np.empty_like(image)  # Create empty array with same shape

    # Swap diagonally (q1<->q4, q2<->q3)
    # q1 -> bottom-right
    swapped[20:, 20:, :] = image[:20, :20, :]
    # q2 -> bottom-left
    swapped[20:, :20, :] = image[:20, 20:, :]
    # q3 -> top-right
    swapped[:20, 20:, :] = image[20:, :20, :]
    # q4 -> top-left
    swapped[:20, :20, :] = image[20:, 20:, :]

    return swapped



def fast_data_generator(file_list, targets, cultivars, batch_size, img_size, band_filter=None):
    num_samples = len(file_list)
    missing_files = []

    def preprocess(file, i, offset):
        try:
            data = np.load(file)

            # Crop and swap quadrants
            if img_size == 40 or img_size == 20:
                data = data[5:-5, 5:-5, :]
            # data = swap_quadrants_fast(data)

            # Filter spectral bands if required
            if band_filter is not None:
                data = data[:, :, band_filter]


            # Expand cultivar info to match spatial dims
            # cultivar = cultivars[offset + i]
            # cultivar_exp = np.broadcast_to(cultivar.reshape(1, 1, -1), (img_size, img_size, cultivar.shape[-1]))

            # Concatenate along the channel axis
            # combined = np.concatenate([data, cultivar_exp], axis=-1)
            combined = data

            return combined, targets[offset + i]

        except FileNotFoundError:
            missing_files.append(file)
            print(f"File not found: {file}. Skipping...")
            return None, None

    while True:
        # tqdm wraps the outer loop to show progress per epoch
        for offset in tqdm(range(0, num_samples, batch_size), desc="Epoch file processing"):
            batch_files = file_list[offset:offset + batch_size]

            # Process each file in the batch
            processed = [preprocess(f, i, offset) for i, f in enumerate(batch_files)]

            # Filter out any None results due to missing files
            processed = [p for p in processed if p[0] is not None]

            if not processed:
                continue

            batch_data, batch_targets = zip(*processed)
            yield np.stack(batch_data), np.array(batch_targets)


# Data generators help with managing memory when using large amounts of data

# def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size, band_filter=None):
#     num_samples = len(file_list)
#     missing_files = [] # List of missing files

#     while True:  # Infinite loop for generator
#         for offset in range(0, num_samples, batch_size):
#             # Load the batch of data from file paths
#             batch_files = file_list[offset: offset + batch_size]
#             batch_data = []
#             batch_targets = []
#             batch_cultivars = []

#             # File loading and handling - ensures model runs if file not found
#             for i, file in enumerate(batch_files):
#                 try:
#                     data = np.load(file)
#                     if img_size == 40 or img_size ==20:
#                         data_reduced = data[5:-5, 5:-5, :] # Remove 5 pixels from each edge
#                         batch_data.append(data_reduced)
#                     else:                    
#                         batch_data.append(data)
#                     batch_targets.append(targets[offset + i])
#                     batch_cultivars.append(cultivars[offset + i])
#                 except FileNotFoundError:
#                     missing_files.append(file)
#                     continue
            

#             # Convert lists to numpy arrays
#             batch_data = np.array(batch_data)  # Shape: (batch_size, 50, 50, 204)
#             batch_targets = np.array(batch_targets)  # Shape: (batch_size,)
#             batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)
            
#             if len(batch_data) == 0:
#                 continue # Skip if no data loaded

#             # Filter some spectral bands if needed
#             if band_filter is not None:

#                 filtered_test = batch_data[:, :, :, band_filter] # % of the features for brix

#                 batch_data = filtered_test
            
                       
#             # Expand cultivar information to match the input data's spatial dimensions
#             expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1) # Adds singleton dimensions to match the input data's spatial dimensions (batchsize, 1, 1, 1, 6)
#             expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)

#             # Concatenate cultivar information with the original data along the last axis
#             combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, 50, 50, 210)


#             # Yield the combined data and targets
#             yield combined_data, batch_targets

#         # After the loop, print and save the missing files
#         if missing_files:
#             print(f"Missing files: {missing_files}")
#             missing_files = []  # Clear the list after saving
    

# In[ ]:


# Assuming input shape is (50, 50, 210)


if band_filter is not None:
    channel_size = np.sum(selected_features_threshold_starch) + 6
    input_shape = (img_size, img_size, channel_size)  
else:
    input_shape = (img_size, img_size, channels) # 204 spectral bands + 6 cultivar onehot encoding



# In[ ]:


# Define Vision Transformer components

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def create_vit_model(
    input_shape,
    patch_size=5,  # Size of patches to extract from the input image
    projection_dim=128,  # Embedding dimension for each patch
    transformer_layers=4,  # Number of Transformer blocks
    num_heads=8,  # Number of attention heads
    mlp_head_units=[256, 128, 64],  # Hidden units in the MLP head
    dropout_rate=0.1,  # Dropout rate
):
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Create patches using Conv2D and Reshape
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
    )(inputs)
    patches = layers.Reshape((-1, projection_dim))(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    
    # Add position embeddings to patch embeddings
    encoded_patches = patches + position_embedding

    attention_outputs = []
    
    # Create Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate,
            output_shape=None  # optional
        )
        # Call the layer and get the outputs
        attention_output, attention_scores = attention_layer(
            x1, x1, return_attention_scores=True
        )
        attention_outputs.append(attention_scores)  # Store attention scores for analysis
        
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
    regression_output = Dense(1, activation="linear", name="regression")(features)

    # # Create the model
    # outputs = {"regression": regression_output}
    # for i, attn in enumerate(attention_outputs):
    #     outputs[f"attention_{i}"] = attn

    model = Model(inputs=inputs, outputs=regression_output)
    return model


# In[ ]:


checkpoint_starch = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_starch_vit.keras", 
                    # monitor="val_regression_mae", 
                    monitor="val_mae",
                    mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)

# 添加早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(
                    # monitor='val_regression_mae',
                    monitor='val_mae',
                    min_delta=0.001,  # 最小变化阈值
                    patience=15,      # 如果验证MAE在15轮内没有改善，则停止训练
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)  # 恢复最佳权重

reduce_lr = ReduceLROnPlateau(
    # monitor='val_regression_mae',
    monitor='val_mae',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)


# In[ ]:

# 使用分布式策略，支持多GPU训练
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     # Create Vision Transformer model
#     model_vit_starch = create_vit_model(
#         input_shape=input_shape,
#         patch_size=5,  # 每个patch的大小
#         projection_dim=64,  # 每个patch的嵌入维度
#         transformer_layers=4,  # Transformer块的数量
#         num_heads=4,  # 注意力头的数量
#         mlp_head_units=[64, 32, 16],  # MLP头部的隐藏单元
#         dropout_rate=0.1,  # Dropout率
#     )
    
#     # Compile the model
#     model_vit_starch.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 使用较小的学习率
#         loss="mean_squared_error", 
#         metrics=["mae"]
#     )

# model_vit_starch.summary()

import pandas as pd

opt_bays_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Bayesian_optimisation_files/'
starch_param = pd.read_pickle(f'{opt_bays_path}bayesian_optimization_results_starch_ViT.pkl')
min_starch = starch_param.loc[starch_param['val_loss'].idxmin()]
min_starch_dict = min_starch.to_dict()
min_starch_dict['mlp_head_units'] = list(map(int, min_starch_dict['mlp_head_units'].split('-')))

num_attention_layers = min_starch_dict['transformer_layers']  # Number of attention layers to monitor

# 使用分布式策略，支持多GPU训练
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Create Vision Transformer model
    model_vit_starch = create_vit_model(
        input_shape=input_shape,
        patch_size=min_starch_dict['patch_size'],  # Adjust based on your image size
        projection_dim=min_starch_dict['projection_dim'],
        transformer_layers=min_starch_dict['transformer_layers'],
        num_heads=min_starch_dict['num_heads'],
        mlp_head_units=min_starch_dict['mlp_head_units'],
        dropout_rate=min_starch_dict['dropout_rate'],
    )
    
    # Compile the model
    model_vit_starch.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 使用较小的学习率
        # loss={"regression": "mean_squared_error"},
        # metrics={"regression": "mae"}
        loss="mean_squared_error",
        metrics=["mae"]
    )

model_vit_starch.summary()


# In[ ]:


# load all_data

if img_size == 30 or img_size == 20:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'


X_train_Starch       = np.load(f'{training_data_path}X_train_all_years_Starch_shuffled.npy')
Y_train_Starch       = np.load(f'{training_data_path}Y_train_all_years_Starch_shuffled.npy')
X_validate_Starch    = np.load(f'{training_data_path}X_validate_all_years_Starch_shuffled.npy')
Y_validate_Starch    = np.load(f'{training_data_path}Y_validate_all_years_Starch_shuffled.npy')
Frmness_encoder_shuffled = np.load(f'{training_data_path}X_train_all_years_Starch_encoder_shuffled.npy')
validate_encoder     = np.load(f'{training_data_path}X_validate_all_years_Starch_encoder_shuffled.npy')


# # Load 2023 data
# if img_size == 30 or img_size == 20:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/NZ2023/'
# elif img_size == 50 or img_size ==40:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/NZ2023/'

# X_train_Starch          = np.load(f'{training_data_path}X_train_NZ2023_Starch_shuffled.npy')
# Y_train_Starch          = np.load(f'{training_data_path}Y_train_NZ2023_Starch_shuffled.npy')
# X_validate_Starch       = np.load(f'{training_data_path}X_validate_NZ2023_Starch_shuffled.npy')
# Y_validate_Starch       = np.load(f'{training_data_path}Y_validate_NZ2023_Starch_shuffled.npy')
# Frmness_encoder_shuffled  = np.load(f'{training_data_path}X_train_NZ2023_Starch_encoder_shuffled.npy')
# validate_encoder        = np.load(f'{training_data_path}X_validate_NZ2023_Starch_encoder_shuffled.npy')

# # # Side A
# if img_size == 30 or img_size == 20:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/'
# elif img_size == 50 or img_size ==40:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/'
# X_train_Starch          = np.load(f'{training_data_path}Side_A_X_train_all_years_Starch_shuffled.npy')
# Y_train_Starch          = np.load(f'{training_data_path}Side_A_Y_train_all_years_Starch_shuffled.npy')
# X_validate_Starch       = np.load(f'{training_data_path}Side_A_X_validate_all_years_Starch_shuffled.npy')
# Y_validate_Starch       = np.load(f'{training_data_path}Side_A_Y_validate_all_years_Starch_shuffled.npy')
# Frmness_encoder_shuffled  = np.load(f'{training_data_path}Side_A_X_train_all_years_Starch_encoder.npy')
# validate_encoder        = np.load(f'{training_data_path}Side_A_X_validate_all_years_Starch_encoder.npy')

# In[ ]:

spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

X_train_Starch = [spectral_path + file for file in X_train_Starch]
X_validate_Starch = [spectral_path + file for file in X_validate_Starch]


# In[ ]:


from datetime import datetime
batch_size = 16  # 批次大小

today = datetime.today().strftime('%Y-%m-%d')

csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}_history_starch_vit.csv", append=True)


# In[ ]:

# 使用数据增强的训练生成器
train_generator = fast_data_generator(X_train_Starch, Y_train_Starch, Frmness_encoder_shuffled, batch_size, img_size=img_size, band_filter=None)
val_generator = fast_data_generator(X_validate_Starch, Y_validate_Starch, validate_encoder, batch_size, img_size=img_size, band_filter=None)


# In[ ]:


# 手动实现的权重衰减回调
weight_decay_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: [
        w.assign(w * (1 - 1e-5)) 
        for w in model_vit_starch.trainable_weights 
        if 'kernel' in w.name
    ]
)

# 训练模型
history = model_vit_starch.fit(
    train_generator,
    epochs=120,  # 增加训练轮次
    steps_per_epoch=len(X_train_Starch) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_validate_Starch) // batch_size,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}_history_starch_vit.csv", append=True), 
        checkpoint_starch,
        early_stopping,
        reduce_lr,
        weight_decay_callback
    ],
    use_multiprocessing=False,
    workers=1,
    max_queue_size=10
)


# In[ ]:


model_vit_starch.save(f'{save_file_path}{Code_run_ID}_model_trained_vit_starch.keras')


# In[ ]:

# 可视化训练历史
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制MAE曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['regression_mae'], label='Training MAE')
plt.plot(history.history['val_regression_mae'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(f"{save_file_path}{Code_run_ID}_training_history_starch.png", dpi=300)
plt.show()

# In[ ]:


del model_vit_starch
gc.collect()

starch_end = datetime.now()
print("总训练时间 Total training time:")
print(starch_end - start_time)
starch_total_time = starch_end - start_time
print(starch_total_time)
print(datetime.now()) 