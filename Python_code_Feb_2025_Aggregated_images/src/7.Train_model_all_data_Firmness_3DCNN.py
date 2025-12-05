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
Code_run_ID = today + f'run_9DCNN_{img_size}px'



# In[ ]:


save_file_path = '/media/2tbdisk3/data/Haidee/Results/'


# In[ ]:


# import os

# os.makedirs(Code_run_ID, exist_ok=True)

# print(f"Directory '{Code_run_ID}' created successfully!")


# In[ ]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    Conv2D,
    MaxPooling2D,
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

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


# 高光谱图像的3D数据生成器 - 包含数据增强

def data_generator_3d(file_list, targets, cultivars, batch_size, img_size, spectral_bands=204, augment=False):
    num_samples = len(file_list)
    missing_files = [] # List of missing files

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
                    
                    # 确保数据有正确的形状
                    if data.shape[2] >= spectral_bands:
                        # 只使用前204个光谱波段
                        data = data[:, :, :spectral_bands]
                    
                    # 数据增强 (仅当augment=True时)
                    if augment and np.random.random() > 0.5:
                        # 随机水平翻转
                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=1)
                        
                        # 随机垂直翻转
                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=0)
                        
                        # 添加微小高斯噪声
                        if np.random.random() > 0.7:
                            noise = np.random.normal(0, 0.02, data.shape)
                            data = data + noise
                            
                        # 随机旋转90度
                        if np.random.random() > 0.8:
                            k = np.random.randint(1, 4)  # 旋转次数
                            data = np.rot90(data, k=k, axes=(0, 1))  # 只在空间维度旋转
                    
                    # 调整数据顺序为 (height, width, channels) -> (channels, height, width)
                    # 3D-CNN期望输入形状为 (batch, depth, height, width, channels)
                    # 在这里，我们将光谱维度作为深度
                    data = np.transpose(data, (2, 0, 1))  # 变为 (spectral_bands, height, width)
                    
                    batch_data.append(data)
                    batch_targets.append(targets[batch_indices[i]])
                    batch_cultivars.append(cultivars[batch_indices[i]])
                except FileNotFoundError:
                    missing_files.append(file)
                    continue
            
            # 如果没有加载任何数据，继续下一批
            if len(batch_data) == 0:
                continue

            # 转换为numpy数组
            batch_data = np.array(batch_data)  # 形状: (batch_size, spectral_bands, height, width)
            batch_targets = np.array(batch_targets)  # 形状: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # 形状: (batch_size, 6)
            
            # 将批次数据重塑为3D-CNN输入格式
            # (batch_size, spectral_bands, height, width) -> (batch_size, spectral_bands, height, width, 1)
            batch_data = np.expand_dims(batch_data, axis=-1)
            
            # 品种信息不需要转换为3D形式，直接传递给模型
            # 返回数据和目标值
            yield [batch_data, batch_cultivars], batch_targets

        # 在循环结束后，打印并保存缺失的文件
        if missing_files:
            print(f"Missing files: {missing_files}")
            missing_files = []  # 清空列表
    

# In[ ]:


# 输入形状 - 3D CNN需要5D输入 (batch, depth, height, width, channels)
# 在我们的例子中，depth = spectral_bands, channels = 1 (单通道)
spectral_bands = 204
spatial_size = img_size
input_shape_3d = (spectral_bands, spatial_size, spatial_size, 1)
input_shape_cultivar = (6,)  # 品种信息的输入形状


# In[ ]:


def create_3d_cnn_model(spectral_input_shape, cultivar_input_shape, filters=(32, 64, 128, 160, 192), dropout_rate=0.3):
    """
    创建增强版3D-CNN模型，添加更多卷积层，增强特征提取能力
    
    参数:
    - spectral_input_shape: 光谱输入的形状 (spectral_bands, height, width, channels)
    - cultivar_input_shape: 品种输入的形状 (通常是 (6,) 表示6个品种)
    - filters: 3D卷积块中的过滤器数量序列
    - dropout_rate: Dropout层的丢弃率
    
    返回:
    - model: 编译好的Keras模型
    """
    # 光谱输入
    spectral_input = Input(shape=spectral_input_shape, name='spectral_input')
    
    # 3D卷积块 1
    x = Conv3D(filters[0], kernel_size=(7, 3, 3), padding='same')(spectral_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 添加残差连接的辅助卷积层
    x_res = Conv3D(filters[0], kernel_size=(1, 1, 1), padding='same')(spectral_input)
    x = Add()([x, x_res])  # 残差连接
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # 3D卷积块 2
    x = Conv3D(filters[1], kernel_size=(5, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 增加一个额外的卷积层，保持相同的空间维度
    x = Conv3D(filters[1], kernel_size=(5, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # 3D卷积块 3
    x = Conv3D(filters[2], kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 增加一个额外的卷积层，保持相同的空间维度
    x = Conv3D(filters[2], kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # 新增 3D卷积块 4
    x = Conv3D(filters[3], kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters[3], kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # 新增 3D卷积块 5 (不进行下采样，保留空间信息)
    x = Conv3D(filters[4], kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # 全局平均池化，提取特征
    x = GlobalAveragePooling3D()(x)
    
    # 品种输入
    cultivar_input = Input(shape=cultivar_input_shape, name='cultivar_input')
    
    # 合并光谱特征和品种信息
    merged = Concatenate()([x, cultivar_input])
    
    # 全连接层 - 增加神经元数量
    x = Dense(512)(merged)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 输出层 - 回归任务
    output = Dense(1, name='firmness_output')(x)
    
    # 创建和编译模型
    model = Model(inputs=[spectral_input, cultivar_input], outputs=output)
    
    return model


# In[ ]:


# 建立检查点和回调函数

checkpoint_firmness = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_firmness_3dcnn.keras", 
                    monitor="val_mae", mode="min", 
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1)

# 早停机制
early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_mae',
                    min_delta=0.0005,  # 最小变化阈值
                    patience=20,      # 从10增加到20，给模型更多时间收敛
                    verbose=1,
                    mode='min',
                    restore_best_weights=True)

# 学习率降低策略
reduce_lr = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=10,      # 保持不变
    min_lr=1e-6,
    verbose=1,
    mode='min'
)

# In[ ]:


# 创建和编译模型
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 创建增强版3D-CNN模型
    # 主要增强点：
    # 1. 增加卷积层数量 - 从3个卷积块增加到5个
    # 2. 每个卷积块中加入多个卷积层
    # 3. 添加残差连接
    # 4. 增加全连接层的神经元数量
    # 5. 减小批次大小以适应更复杂的模型
    model_3dcnn_firmness = create_3d_cnn_model(
        spectral_input_shape=input_shape_3d,
        cultivar_input_shape=input_shape_cultivar,
        filters=(32, 64, 128, 160, 192),
        dropout_rate=0.3
    )
    
    # 使用Huber Loss代替MSE，对异常值更稳健
    huber_loss = Huber(delta=1.0)
    
    # 优化器 - 减小学习率以适应更深的网络
    optimizer = Adam(
        learning_rate=0.0005,  # 将学习率从0.001降低到0.0005
        beta_1=0.9, 
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        clipnorm=1.0  # 梯度裁剪
    )
    
    # 编译模型
    model_3dcnn_firmness.compile(
        optimizer=optimizer,
        loss=huber_loss,
        metrics=["mae"]
    )

# 打印模型概要
model_3dcnn_firmness.summary()


# In[ ]:


# 加载所有数据

if img_size == 30 or img_size == 20:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'


X_train_Firmness       = np.load(f'{training_data_path}X_train_all_years_Firmness_shuffled.npy')
Y_train_Firmness       = np.load(f'{training_data_path}Y_train_all_years_Firmness_shuffled.npy')
X_validate_Firmness    = np.load(f'{training_data_path}X_validate_all_years_Firmness_shuffled.npy')
Y_validate_Firmness    = np.load(f'{training_data_path}Y_validate_all_years_Firmness_shuffled.npy')
Firmness_encoder_shuffled = np.load(f'{training_data_path}X_train_all_years_Firmness_encoder_shuffled.npy')
validate_encoder     = np.load(f'{training_data_path}X_validate_all_years_Firmness_encoder_shuffled.npy')


# In[ ]:


spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

X_train_Firmness = [spectral_path + file for file in X_train_Firmness]
X_validate_Firmness = [spectral_path + file for file in X_validate_Firmness]


# In[ ]:


batch_size = 8  # 将批次大小从16减小到8，以适应更深的模型结构

csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}_history_firmness_3dcnn.csv", append=True)


# In[ ]:


# 手动实现的权重衰减回调
weight_decay_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: [
        w.assign(w * (1 - 1e-5)) 
        for w in model_3dcnn_firmness.trainable_weights 
        if 'kernel' in w.name
    ]
)

# 使用数据增强的训练生成器，验证集不使用增强
train_generator = data_generator_3d(X_train_Firmness, Y_train_Firmness, Firmness_encoder_shuffled, batch_size, img_size=img_size, augment=True)
val_generator = data_generator_3d(X_validate_Firmness, Y_validate_Firmness, validate_encoder, batch_size, img_size=img_size, augment=False)


# In[ ]:


# 训练模型
epochs = 150  # 从100增加到150
history = model_3dcnn_firmness.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=len(X_train_Firmness) // batch_size,
    validation_data=val_generator,
    validation_steps=len(X_validate_Firmness) // batch_size,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}_history_firmness_3dcnn.csv", append=True), 
        checkpoint_firmness,
        early_stopping,
        reduce_lr,
        weight_decay_callback
    ],
)


# In[ ]:


# 可视化训练历史
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

# 绘制MAE曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('平均绝对误差(MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(f"{save_file_path}{Code_run_ID}_training_history.png", dpi=300)
plt.show()


# In[ ]:


# 保存最终模型
model_3dcnn_firmness.save(f'{save_file_path}{Code_run_ID}_model_trained_3dcnn_firmness.keras')


# In[ ]:


# 释放资源
del model_3dcnn_firmness
gc.collect()

firmness_end = datetime.now()
print("总训练时间 Total training time:")
print(firmness_end - start_time)
firmness_total_time = firmness_end - start_time
print(firmness_total_time)
print(datetime.now()) 