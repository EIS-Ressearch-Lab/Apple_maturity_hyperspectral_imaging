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


# 尝试启用混合精度训练 - 兼容多个TensorFlow版本 # "Attempt to enable mixed precision training – compatible with multiple TensorFlow versions."
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


# 高光谱图像的3D数据生成器 - 包含数据增强 # "3D data generator for hyperspectral images – includes data augmentation."

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
                    
                    # 确保数据有正确的形状 # Make sure have correct shape
                    if data.shape[2] >= spectral_bands:
                        # 只使用前204个光谱波段
                        data = data[:, :, :spectral_bands]
                    
                    # 数据增强 (仅当augment=True时) # Data enhancement (only when auent = True)
                    if augment and np.random.random() > 0.5:
                        # 随机水平翻转 # Random horizontal flip
                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=1)
                        
                        # 随机垂直翻转 # Random vertical flip
                        if np.random.random() > 0.5:
                            data = np.flip(data, axis=0)
                        
                        # 添加微小高斯噪声 # Add a tiny Gaussian sound absorption
                        if np.random.random() > 0.7:
                            noise = np.random.normal(0, 0.02, data.shape)
                            data = data + noise
                            
                        # 随机旋转90度 # Randomly rotate 90 degrees
                        if np.random.random() > 0.8:
                            k = np.random.randint(1, 4)  # 旋转次数 #Number of rotations
                            data = np.rot90(data, k=k, axes=(0, 1))  # 只在空间维度旋转 #Rotate only in spatial dimension
                    
                    # 调整数据顺序为 (height, width, channels) -> (channels, height, width) # Adjust the data order from (height, width, channels) to (channels, height, width)
                    # 3D-CNN期望输入形状为 (batch, depth, height, width, channels) # 3D CNN expects input shape as (batch, depth, height, width, channels)
                    # 在这里，我们将光谱维度作为深度 # Here, we treat the spectral dimension as depth
                    data = np.transpose(data, (2, 0, 1))  # 变为 (spectral_bands, height, width) # becomes (spectral_bands, height, width)
                    
                    batch_data.append(data)
                    batch_targets.append(targets[batch_indices[i]])
                    batch_cultivars.append(cultivars[batch_indices[i]])
                except FileNotFoundError:
                    missing_files.append(file)
                    continue
            
            # 如果没有加载任何数据，继续下一批 # If no data loaded, continue to the next batch
            if len(batch_data) == 0:
                continue

            # 转换为numpy数组 # Convert to numpy arrays
            batch_data = np.array(batch_data)  # 形状: (batch_size, spectral_bands, height, width)
            batch_targets = np.array(batch_targets)  # 形状: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # 形状: (batch_size, 6)
            
            # 将批次数据重塑为3D-CNN输入格式 # Model batch data into 3D-CNN input format
            # (batch_size, spectral_bands, height, width) -> (batch_size, spectral_bands, height, width, 1)
            batch_data = np.expand_dims(batch_data, axis=-1)
            
            # 品种信息不需要转换为3D形式，直接传递给模型 # Cultivar information does not need to be converted to 3D format, directly passed to the model
            # 返回数据和目标值 # Return data and target values
            yield [batch_data, batch_cultivars], batch_targets

        # 在循环结束后，打印并保存缺失的文件 # After the loop ends, print and save the missing files
        if missing_files:
            print(f"Missing files: {missing_files}")
            missing_files = []  # 清空列表 # Clear the list after printing
    

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
    output = Dense(1, name='brix_output')(x)
    
    # 创建和编译模型
    model = Model(inputs=[spectral_input, cultivar_input], outputs=output)
    
    return model


# In[ ]:


# 建立检查点和回调函数

checkpoint_brix = tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{save_file_path}{Code_run_ID}_model_file_brix_3dcnn.keras", 
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
    model_3dcnn_brix = create_3d_cnn_model(
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
    model_3dcnn_brix.compile(
        optimizer=optimizer,
        loss=huber_loss,
        metrics=["mae"]
    )

# 打印模型概要
model_3dcnn_brix.summary()


# # In[ ]:


# # 加载所有数据 # Load all data

# if img_size == 30 or img_size == 20:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
# elif img_size == 50 or img_size ==40:
#     training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'


# X_train_brix          = np.load(f'{training_data_path}X_train_all_years_Brix_shuffled.npy')
# Y_train_brix          = np.load(f'{training_data_path}Y_train_all_years_Brix_shuffled.npy')
# X_validate_brix       = np.load(f'{training_data_path}X_validate_all_years_Brix_shuffled.npy')
# Y_validate_brix       = np.load(f'{training_data_path}Y_validate_all_years_Brix_shuffled.npy')
# Frmness_encoder_shuffled  = np.load(f'{training_data_path}X_train_all_years_Brix_encoder_shuffled.npy')
# validate_encoder        = np.load(f'{training_data_path}X_validate_all_years_Brix_encoder_shuffled.npy')


# # In[ ]:


# spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'

# X_train_brix = [spectral_path + file for file in X_train_brix]
# X_validate_brix = [spectral_path + file for file in X_validate_brix]


# # In[ ]:


# from datetime import datetime
# batch_size = 8  # 将批次大小从16减小到8，以适应更深的模型结构 # Reduce the batch size from 16 to 8 to accommodate a deeper model architecture.

# today = datetime.today().strftime('%Y-%m-%d')

# csv_logger = CSVLogger(f"{save_file_path}{Code_run_ID}_history_brix_3dcnn.csv", append=True)


# # In[ ]:


# # 手动实现的权重衰减回调
# weight_decay_callback = tf.keras.callbacks.LambdaCallback(
#     on_epoch_end=lambda epoch, logs: [
#         w.assign(w * (1 - 1e-5)) 
#         for w in model_3dcnn_brix.trainable_weights 
#         if 'kernel' in w.name
#     ]
# )

# # 使用数据增强的训练生成器，验证集不使用增强
# train_generator = data_generator_3d(X_train_brix, Y_train_brix, Frmness_encoder_shuffled, batch_size, img_size=img_size, augment=True)
# val_generator = data_generator_3d(X_validate_brix, Y_validate_brix, validate_encoder, batch_size, img_size=img_size, augment=False)


# # In[ ]:


# # 训练模型
# epochs = 150  # 从100增加到150
# history = model_3dcnn_brix.fit(
#     train_generator,
#     epochs=epochs,
#     steps_per_epoch=len(X_train_brix) // batch_size,
#     validation_data=val_generator,
#     validation_steps=len(X_validate_brix) // batch_size,
#     callbacks=[
#         tf.keras.callbacks.CSVLogger(f"{save_file_path}{Code_run_ID}_history_brix_3dcnn.csv", append=True), 
#         checkpoint_brix,
#         early_stopping,
#         reduce_lr,
#         weight_decay_callback
#     ],
# )


# # In[ ]:


# # 保存最终模型 Save the final model
# model_3dcnn_brix.save(f'{save_file_path}{Code_run_ID}_model_trained_3dcnn_brix.keras')


# # In[ ]:


# # 可视化训练历史
# import matplotlib.pyplot as plt

# plt.figure(figsize=(16, 6))

# # 绘制损失曲线
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='训练损失')
# plt.plot(history.history['val_loss'], label='验证损失')
# plt.title('模型损失')
# plt.xlabel('Epoch')
# plt.ylabel('损失')
# plt.legend()

# # 绘制MAE曲线
# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='训练MAE')
# plt.plot(history.history['val_mae'], label='验证MAE')
# plt.title('平均绝对误差(MAE)')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()

# plt.tight_layout()
# plt.savefig(f"{save_file_path}{Code_run_ID}_training_history.png", dpi=300)
# plt.show()


# # In[ ]:


# # 释放资源
# del model_3dcnn_brix
# gc.collect()

# brix_end = datetime.now()
# print("总训练时间 Total training time:")
# print(brix_end - start_time)
# brix_total_time = brix_end - start_time
# print(brix_total_time)
# print(datetime.now()) 


# 加载测试数据
def load_test_data(test_data_path):
    # 加载测试数据
    X_test_brix = np.load(f'{test_data_path}X_test_all_years_Brix_shuffled.npy')
    Y_test_brix = np.load(f'{test_data_path}Y_test_all_years_Brix_shuffled.npy')
    test_encoder = np.load(f'{test_data_path}X_test_all_years_Brix_encoder_shuffled.npy')
    
    # 添加路径前缀
    spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'
    X_test_brix = [spectral_path + file for file in X_test_brix]
    
    return X_test_brix, Y_test_brix, test_encoder


# 自定义计算评估指标的函数
def calculate_metrics(y_true, y_pred):
    # 计算MSE
    mse = np.mean((y_true - y_pred) ** 2)
    # 计算RMSE
    rmse = np.sqrt(mse)
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    # 计算R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mse, rmse, mae, r2


# 对测试集进行预测并可视化 # Make predictions on the test set and visualize
def predict_and_visualize(model, X_test, Y_test, test_encoder, img_size, save_dir):
    print("Preparing test data; 准备测试数据...")
    test_data, test_targets = data_generator_w_cultivar(X_test, Y_test, test_encoder, len(X_test), img_size)
    
    print("Make predicitons; 进行预测...")
    predictions = model.predict(test_data)
    predictions = predictions.flatten()  # 将预测结果展平为一维数组 Flatten the prediction results into a one-dimensional array
    
    # 计算评估指标 # Calculate evaluation metrics
    mse, rmse, mae, r2 = calculate_metrics(test_targets, predictions)
    
    # 打印评估结果 Print evaluation results
    print(f"\n Test set evaluation results :") #; 测试集评估结果
    print(f"Mean square error (MSE): {mse:.4f}") # ;均方误差
    print(f"Root mean square error (RMSE): {rmse:.4f}") # 均方根误差
    print(f"Mean absolute error (MAE): {mae:.4f}") # 平均绝对误差
    print(f"coefficient of determination (R²): {r2:.4f}") #决定系数
    
    # 创建散点图 # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(test_targets, predictions, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.title('Hybrid Models: Actual vs Predicted Values') # 混合模型：实际值 vs 预测值
    plt.xlabel('Actual (Brix)') #实际糖度
    plt.ylabel('Predicted (Brix)') # 预测糖度
    plt.grid(True)
    
    # 添加评估指标文本 # Add evaluation metrics text
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_hybrid_test_predictions.png', dpi=300)
    plt.show()
    
    # 创建误差分布直方图 # Create error distribution histogram
    errors = predictions - test_targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, density=True)
    
    # 添加核密度估计曲线 # Add kernel density estimation curve
    x = np.linspace(min(errors), max(errors), 100)
    kde = stats.gaussian_kde(errors)
    plt.plot(x, kde(x), 'r-', linewidth=2)
    
    plt.title('Hybrid model: Prediction Error Distribution') # 混合模型：预测误差分布 # Hybrid model: Prediction Error Distribution
    plt.xlabel('Prediction Error') #预测误差 # Prediction Error
    plt.ylabel('Frequency') # 频率 # 
    plt.grid(True)
    plt.axvline(x=0, color='r', linestyle='--')
    
   
    # 添加误差统计信息 # Add error statistics text
    plt.text(0.05, 0.95, f'Mean error: {np.mean(errors):.4f}\nStandard deviation: {np.std(errors):.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_hybrid_error_distribution.png', dpi=300)
    plt.show()
    
    # 保存预测结果和实际值到CSV文件
    results_df = pd.DataFrame({
        'Actual': test_targets,
        'Predicted': predictions,
        'Error': errors
    })
    results_df.to_csv(f'{save_dir}/brix_hybrid_prediction_results.csv', index=False)
    
    return results_df, mse, rmse, mae, r2


# 可视化不同品种的预测性能 # Visualize prediction performance by cultivar
def visualize_by_cultivar(results_df, test_encoder, save_dir):
    # 创建品种名称列表 # Create cultivar names list
    cultivar_names = ['Braeburn', 'Cox', 'Fuji','Gala', 'Golden Delicious', 'Jazz', 'Other']
    
    # 获取每个样本的品种（取每个独热编码的最大索引） # Get cultivar index for each sample (take the index of the maximum value in one-hot encoding)
    cultivar_indices = np.argmax(test_encoder, axis=1)
    
    # 将品种索引添加到结果DataFrame # Add cultivar indices to the results DataFrame
    results_df['Cultivar_Index'] = cultivar_indices
    results_df['Cultivar'] = [cultivar_names[idx] for idx in cultivar_indices]
    
    # 按品种分组计算评估指标 # Group by cultivar and calculate evaluation metrics
    cultivar_metrics = []
    for i, name in enumerate(cultivar_names):
        cultivar_data = results_df[results_df['Cultivar_Index'] == i]
        if len(cultivar_data) > 0:  # 确保有该品种的数据 # Ensure there is data for this cultivar
            actual = cultivar_data['Actual'].values
            predicted = cultivar_data['Predicted'].values
            mse, rmse, mae, r2 = calculate_metrics(actual, predicted)
            count = len(cultivar_data)
            cultivar_metrics.append({
                'Cultivar': name,
                'Count': count,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
    
    # 创建品种评估指标DataFrame # Create cultivar metrics DataFrame
    cultivar_metrics_df = pd.DataFrame(cultivar_metrics)
    cultivar_metrics_df.to_csv(f'{save_dir}/brix_hybrid_cultivar_metrics.csv', index=False)
    
    # 打印品种评估指标 # Print cultivar metrics
    print("\n Evaluation indicators for each variety; 各品种评估指标:")
    print(cultivar_metrics_df)
    
    # 可视化各品种的MAE # Visualize MAE for each cultivar
    plt.figure(figsize=(12, 6))
    
    # 创建条形图 # Create bar chart
    x = range(len(cultivar_metrics_df))
    plt.bar(x, cultivar_metrics_df['MAE'], width=0.6)
    plt.xticks(x, cultivar_metrics_df['Cultivar'], rotation=45)
    plt.title('Hybrid model: Mean absolute error for each apple cultivar (MAE)') #混合模型：各苹果品种的平均绝对误差
    plt.xlabel('Cultivar') #苹果品种
    plt.ylabel('MAE')
    plt.grid(True, axis='y')
    
    # 在每个柱子上添加样本数量 # Add sample count on top of each bar
    for i, row in enumerate(cultivar_metrics_df.itertuples()):
        plt.text(i, row.MAE + 0.02, f'n={row.Count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_hybrid_cultivar_mae.png', dpi=300)
    plt.show()
    
    # 创建各品种的散点图 # Create scatter plot for each cultivar
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(cultivar_names):
        cultivar_data = results_df[results_df['Cultivar_Index'] == i]
        if len(cultivar_data) > 0:  # 确保有该品种的数据 # Ensure there is data for this cultivar
            plt.scatter(cultivar_data['Actual'], cultivar_data['Predicted'], 
                       label=f'{name} (n={len(cultivar_data)})', alpha=0.7)
    
    plt.plot([min(results_df['Actual']), max(results_df['Actual'])], 
             [min(results_df['Actual']), max(results_df['Actual'])], 'k--')
    plt.title('Hybrid model: Actual vs Predicted Values by Cultivar') #混合模型：各品种实际值 vs 预测值 # 
    plt.xlabel('Actual (Brix)') #实际糖度 # 
    plt.ylabel('Predicted (Brix)') #预测糖度
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_hybrid_cultivar_predictions.png', dpi=300)
    plt.show()
    
    return cultivar_metrics_df


# 主函数 # Main function
def main():
    print("Start visualistion of hybrix model results; 开始混合模型结果可视化...") 
    
    # 可视化训练历史 # Visualize training history
    print("\n1. Visualize training history; 可视化训练历史")
    best_epoch, best_val_mae = visualize_training_history(history_path, results_dir)
    
    # 加载模型 # Load the model
    print(f"\n2. Loaded model; 加载模型: {model_path}")
    try:
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        print(f"Error loading model; 加载模型时出错: {e}")
        return
    
    # 加载测试数据 # Load test data
    print("\n3. Load test data; 加载测试数据") 

    if img_size == 30 or img_size == 20:
        training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
    elif img_size == 50 or img_size ==40:
        training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

    try:
        X_test, Y_test, test_encoder = load_test_data(training_data_path)
        print(f"Number of test samples; 测试样本数量: {len(X_test)}") # Number of test samples
    except Exception as e:
        print(f"Error loading test data; 加载测试数据时出错: {e}") 
        return
    
    # 预测和评估
    print("\n4. Predict and evaluate on the test set; 对测试集进行预测和评估")
    try:
        results_df, mse, rmse, mae, r2 = predict_and_visualize(model, X_test, Y_test, test_encoder, img_size, results_dir)
    except Exception as e:
        print(f"Errors in prediction and evaluation; 预测和评估时出错: {e}")
        return
    
    # 按品种可视化
    print("\n5. Analyse prediction performance by species; 按品种分析预测性能")
    try:
        cultivar_metrics_df = visualize_by_cultivar(results_df, test_encoder, results_dir)
    except Exception as e:
        print(f"Error when analyzing by species; 按品种分析时出错: {e}")
        return
    
    print(f"\nAll results have been saved to; 所有结果已保存到 '{results_dir}' directory; 目录")


if __name__ == "__main__":
    main() 