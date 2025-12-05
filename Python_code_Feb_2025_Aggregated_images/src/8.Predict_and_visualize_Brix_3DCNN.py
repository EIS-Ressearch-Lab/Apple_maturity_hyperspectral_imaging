#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gc
import glob
import sys
import re
from scipy import stats

img_size = 50


# 设置随机种子以确保结果可重现 # set random seed for reproducible results
np.random.seed(42)
tf.random.set_seed(42)

results_path = '/media/2tbdisk3/data/Haidee/Results' 

print("================ Start Prediction ================") #开始预测
print(f"TensorFlow version: {tf.__version__}") # TensorFlow version 版本
print(f"Current working directory: {os.getcwd()}") # 当前工作目录 Get current working directory

# 查找最新的模型文件 # Find the latest model files
def find_latest_model(model_dir='/media/2tbdisk3/data/Haidee/Results'):
    model_files = glob.glob(f"{model_dir}/*model_file_brix_3dcnn.keras")
    if not model_files:
        print(f"Error: No model was found in directory{model_dir}") #print(f"错误: 在 {model_dir} 目录中没有找到模型文件")
        return None
    
    # 按修改时间排序 # Sort by date modified
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model}") #找到最新模型
    return latest_model

# 自动查找最新模型 # Automatically find the latest model
model_path = find_latest_model()

# Get run_no
match = re.search(r'run_\d+', model_path)
run_no = match.group(0) if match else None
print(run_no)

pattern = f'{results_path}/*{run_no}*_history_brix_3dcnn.csv'
history_path_matches = glob.glob(pattern)
history_path = history_path_matches[0]


if not model_path:
    # 使用预设路径作为后备 # Use a preset path as a fallback
    model_path = '/media/2tbdisk3/data/Haidee/Results/2025-05-27run_9DCNN_50px_model_file_brix_3dcnn.keras'
    print(f"Use the preset model path: {model_path}") #使用预设模型路径

# 检查模型文件是否存在 # Check if the model file exists
if not os.path.exists(model_path):
    print(f"错误: 模型文件 {model_path} 不存在! Error: Model file does not exist")
    print("请检查路径或训练模型后再尝试; Please check the path or train the model before trying again")
    sys.exit(1)

# 加载训练好的模型 # Load the trained model
try:
    print(f"Load model 加载模型: {model_path}")  
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully 模型加载成功!")
    # 打印模型结构摘要 # Print model summary
    model.summary()
except Exception as e:
    print(f"Failed to load model 加载模型失败: {str(e)}")
    sys.exit(1)

# 获取模型输入信息 # Get model input information
input_shapes = [input.shape for input in model.inputs]
print(f"Model input shapes; 模型输入形状: {input_shapes}")

# 加载测试数据 # Load test data


if img_size == 30 or img_size == 20:
    test_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
elif img_size == 50 or img_size ==40:
    test_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

try:
    print(f"从 {test_data_path} 加载测试数据; load test data from path")
    X_test_brix = np.load(f'{test_data_path}X_test_all_years_Brix_shuffled.npy')
    Y_test_brix = np.load(f'{test_data_path}Y_test_all_years_Brix_shuffled.npy')
    test_encoder = np.load(f'{test_data_path}X_test_all_years_Brix_encoder_shuffled.npy')
    
    print(f" Test data; 测试数据加载成功:")
    print(f"X_test_brix: {X_test_brix.shape}, {type(X_test_brix[0])}")
    print(f"Y_test_brix: {Y_test_brix.shape}")
    print(f"test_encoder: {test_encoder.shape}")
    
    # 确保品种编码为float32类型
    if test_encoder.dtype != np.float32:
        test_encoder = test_encoder.astype(np.float32)
        print("Ensure cultivar encoding is converted to float32 type") # 品种编码已转换为float32类型 
except Exception as e:
    print(f"Load test data failed: {str(e)}") #加载测试数据失败 
    sys.exit(1)

# 添加文件路径前缀 # Add spectral data file path prefix
spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'
X_test_brix = [spectral_path + file for file in X_test_brix]

# 检查第一个文件是否可访问
if len(X_test_brix) > 0:
    first_file = X_test_brix[0]
    if not os.path.exists(first_file):
        print(f" Warning: Cannot access the first file {first_file}") # 警告: 无法访问第一个文件 # 
        print("请检查文件路径是否正确或文件是否存在; Please check if the file path is correct or if the file exists")
    else:
        print(f"文件路径检查通过，可以访问; File path check passed, can access the file: {first_file}") # 

# 数据生成器函数 # Data generator function for 3D CNN prediction
def data_generator_3d_predict(file_list, cultivars, batch_size=4, img_size=50, spectral_bands=204):
    num_samples = len(file_list)
    print(f"数据生成器将处理 {num_samples} 个样本，批次大小为 {batch_size}")
    
    for offset in range(0, num_samples, batch_size):
        end_idx = min(offset + batch_size, num_samples)
        batch_indices = list(range(offset, end_idx))
        batch_files = [file_list[i] for i in batch_indices]
        batch_data = []
        batch_cultivars = []
        valid_indices = []
        
        # 诊断信息 # Diagnostic information
        if offset % 20 == 0:
            print(f"正在处理批次; Processing batches {offset//batch_size + 1}/{(num_samples-1)//batch_size + 1}") 
        
        for i, file in enumerate(batch_files):
            try:
                data = np.load(file)
                
                # 检查数据形状 # Check data shape
                if offset == 0 and i == 0:
                    print(f"首个样本原始形状; The original shape of the first sample: {data.shape}")
                
                # 确保数据有正确的形状 # Ensure data has the correct shape
                if data.shape[2] >= spectral_bands:
                    data = data[:, :, :spectral_bands]
                
                # 调整数据顺序为 (height, width, channels) -> (channels, height, width) # Adjust the data order from (height, width, channels) to (channels, height, width)
                data = np.transpose(data, (2, 0, 1))
                batch_data.append(data)
                
                # 使用传入的cultivars数组直接获取对应索引的编码 # Use the provided cultivars array to get the corresponding index encoding directly
                batch_cultivars.append(cultivars[offset + i])
                valid_indices.append(offset + i)
                
            except FileNotFoundError:
                print(f"找不到文件; File not found: {file}")
                continue
            except Exception as e:
                print(f"处理文件; Error processing {file} 时出错; error : {str(e)}")
                continue
        
        if len(batch_data) == 0:
            print(f"警告: 批次; Warning: Batch offset {offset//batch_size + 1} 没有有效数据，跳过; has no valid data, skipping")
            continue
            
        # 转换为numpy数组 # Convert to numpy arrays
        batch_data = np.array(batch_data)
        batch_cultivars = np.array(batch_cultivars)
        
        # 打印形状信息  # Print shape information
        if offset == 0:
            print(f"batch_data_shape; 批次数据形状:")
            print(f"  - spectral_data; 光谱数据: {batch_data.shape}")
            print(f"  - cultivar_data; 品种数据: {batch_cultivars.shape}")
            print(f"  - first 5 samples of cultivar data; 品种数据前5个样本: {batch_cultivars[:5]}")
        
        # 重塑为3D-CNN输入格式 # Reshape to 3D-CNN input format
        batch_data = np.expand_dims(batch_data, axis=-1)
        
        if offset == 0:
            print(f"Shape after expanding dimension; 扩展维度后形状: {batch_data.shape}")
        
        yield [batch_data, batch_cultivars], valid_indices

print("开始预测...; Start prediction...") #开始预测 

# 预测 # initialize lists to store predictions and actual values
predictions = []
actual_values = []
failed_batches = 0
total_batches = 0

# 使用生成器进行预测 # Use the generator for prediction
batch_size = 4  # 减小批次大小以减少内存使用 # Reduce batch size to reduce memory usage
for [spectral_data, cultivar_data], valid_indices in data_generator_3d_predict(X_test_brix, test_encoder, batch_size=batch_size, img_size=img_size):
    total_batches += 1
    try:
        # 确保输入数据类型正确 # Ensure input data types are correct
        if spectral_data.dtype != np.float32:
            spectral_data = spectral_data.astype(np.float32)
        if cultivar_data.dtype != np.float32:
            cultivar_data = cultivar_data.astype(np.float32)
        
        # 确保正确传递两个输入 # Ensure both inputs are passed correctly
        batch_predictions = model.predict([spectral_data, cultivar_data], verbose=0)
        
        # 使用有效索引获取实际值 # Use valid indices to get actual values
        batch_actual = Y_test_brix[valid_indices]
        
        predictions.extend(batch_predictions)
        actual_values.extend(batch_actual)
        
        # 打印进度 # Print progress
        print(f"Processed; 已处理 {len(predictions)}/{len(X_test_brix)} samples; 个样本")
    except Exception as e:
        failed_batches += 1
        print(f"Batch; 批次 {total_batches} Prediction failed; 预测失败: {str(e)}")
        print(f"Spectral data shape; 光谱数据形状: {spectral_data.shape}, Cultivar data shape; 品种数据形状: {cultivar_data.shape}")
        
        # 如果连续失败太多，退出 # If too many consecutive failures, exit
        if failed_batches >= 5:
            print("Too many consecutive failures, abort prediction; 连续失败太多，中止预测")
            break

# 检查是否有足够的预测结果 # Check if there are enough prediction results
if len(predictions) == 0:
    print("Error: No successful prediction results; 错误: 没有成功的预测结果!")
    sys.exit(1)

print(f"Prediction completed, a total of ; 预测完成，共有 {len(predictions)} samples; 个样本")
print(f"failed batch; 失败批次: {failed_batches}/{total_batches}")






# # 转换为numpy数组 # Convert to numpy arrays
# predictions = np.array(predictions)
# actual_values = np.array(actual_values)

# print(f"预测值形状: {predictions.shape}")
# print(f"实际值形状: {actual_values.shape}")

# # 计算评估指标 # Calculate evaluation metrics
# mae = mean_absolute_error(actual_values, predictions)
# mse = mean_squared_error(actual_values, predictions)
# rmse = np.sqrt(mse)
# r2 = r2_score(actual_values, predictions)

# print("Evaluation metrics; 评估指标:")
# print(f"平均绝对误差 (MAE): {mae:.4f}")
# print(f"均方根误差 (RMSE): {rmse:.4f}")
# print(f"决定系数 (R²): {r2:.4f}")

# # 创建结果目录 # Create results directory
results_dir = f'/media/2tbdisk3/data/Haidee/Results/Predictions/{run_no}'
os.makedirs(results_dir, exist_ok=True)
print(f"Save results to directory; 将结果保存到目录: {results_dir}")

# # 保存预测结果到CSV
# results_df = pd.DataFrame({
#     'Actual': actual_values, #实际值
#     'Predicted': predictions.flatten(), #预测值
#     'Absolute Error': np.abs(actual_values - predictions.flatten()) # 误差
# })
# results_df.to_csv(f'{results_dir}/{run_no}_brix_prediction_results.csv', index=False)
# print(f"Results have been saved to; 结果已保存到 {results_dir}/brix_prediction_results.csv")



# # 可视化 # Visualize the results
# print("Create graphs; 创建可视化图表...")
# plt.figure(figsize=(15, 10))

# # 1. 预测值vs实际值散点图 # Scatter plot of predicted vs actual values
# plt.subplot(2, 2, 1)
# plt.scatter(actual_values, predictions, alpha=0.5)
# plt.plot([actual_values.min(), actual_values.max()], 
#          [actual_values.min(), actual_values.max()], 
#          'r--', lw=2)
# plt.xlabel('Actual values') # 实际值 # 
# plt.ylabel('Predicted values') #预测值 
# plt.title('Predicted ​​vs Actual Values') #预测值 vs 实际值 #
# plt.grid(True)

# # 2. 误差分布直方图 # Histogram of prediction errors
# plt.subplot(2, 2, 2)
# sns.histplot(results_df['Absolute Error'], bins=30) #误差
# plt.xlabel('Absolute error') # 绝对误差 # Absolute error
# plt.ylabel('Frequency') #频数  # Frequency
# plt.title('Prediction Error Distribution') #预测误差分布 # Prediction Error Distribution
# plt.grid(True)

# # 3. 预测误差箱线图 # Box plot of prediction errors
# plt.subplot(2, 2, 3)
# sns.boxplot(y=results_df['Absolute Error']) #误差
# plt.ylabel('Absolute error') #绝对误差
# plt.title('Prediction Error Box Plot') # 预测误差箱线图 # Prediction Error Box Plot
# plt.grid(True)

# # 4. 预测值vs误差散点图 # Scatter plot of predicted values vs absolute errors
# plt.subplot(2, 2, 4)
# plt.scatter(predictions, results_df['Absolute Error'], alpha=0.5)
# plt.xlabel('Predicted values') # 预测值 # 
# plt.ylabel('Absolute error') #  # 绝对误差
# plt.title('Predicted Values vs Absolute Error') # 预测值 vs 误差
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(f'{results_dir}/{run_no}_brix_prediction_visualization.png', dpi=300, bbox_inches='tight')
# print(f"Graphs saved to; 可视化图表已保存到 {results_dir}/{run_no}_brix_prediction_visualization.png")

# # 保存评估指标 # Save evaluation metrics to CSV
# metrics_df = pd.DataFrame({
#     'Metrics': ['MAE', 'RMSE', 'R²'], # 指标 # Metrics
#     'Values': [mae, rmse, r2] # 值 # Values
# })
# metrics_df.to_csv(f'{results_dir}/{run_no}_brix_evaluation_metrics.csv', index=False)
# print(f"Evaluation metrics have been saved to; 评估指标已保存到 {results_dir}/{run_no}_brix_evaluation_metrics.csv")



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

# 对测试集进行预测并可视化
def visualize(predictions, actual_values, save_dir):
    
    print("进行预测...")
    
    predictions = np.array(predictions)  # 将预测结果展平为一维数组
    actual_values = np.array(actual_values)  # 将实际值展平为一维数组
    predictions = np.array(predictions).ravel() #flatten arrays
    actual_values = np.array(actual_values).ravel() #flatten arrays
    test_targets = actual_values

    # 计算评估指标 # Calculate evaluation metrics
    mse, rmse, mae, r2 = calculate_metrics(actual_values, predictions)
    
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
    plt.title('3dCNN Models: Actual vs Predicted Values') # 混合模型：实际值 vs 预测值
    plt.xlabel('Actual (Brix)') #实际糖度
    plt.ylabel('Predicted (Brix)') # 预测糖度
    plt.grid(True)
    
    # 添加评估指标文本 # Add evaluation metrics text
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_3dCNN_test_predictions.png', dpi=300)
    plt.show()
    
    # 创建误差分布直方图 # Create error distribution histogram
    errors = predictions - test_targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, density=True)
    
    # 添加核密度估计曲线 # Add kernel density estimation curve
    x = np.linspace(min(errors), max(errors), 100)
    kde = stats.gaussian_kde(errors)
    plt.plot(x, kde(x), 'r-', linewidth=2)
    
    plt.title('3dCNN model: Prediction Error Distribution') # 混合模型：预测误差分布 # 3dCNN model: Prediction Error Distribution
    plt.xlabel('Prediction Error') #预测误差 # Prediction Error
    plt.ylabel('Frequency') # 频率 # 
    plt.grid(True)
    plt.axvline(x=0, color='r', linestyle='--')
    
   
    # 添加误差统计信息 # Add error statistics text
    plt.text(0.05, 0.95, f'Mean error: {np.mean(errors):.4f}\nStandard deviation: {np.std(errors):.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_3dCNN_error_distribution.png', dpi=300)
    plt.show()
    
    # 保存预测结果和实际值到CSV文件
    results_df = pd.DataFrame({
        'Actual': test_targets,
        'Predicted': predictions,
        'Error': errors
    })
    results_df.to_csv(f'{save_dir}/brix_prediction_results.csv', index=False)
    
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
    cultivar_metrics_df.to_csv(f'{save_dir}/brix_3dCNN_cultivar_metrics.csv', index=False)
    
    # 打印品种评估指标 # Print cultivar metrics
    print("\n Evaluation indicators for each variety; 各品种评估指标:")
    print(cultivar_metrics_df)
    
    # 可视化各品种的MAE # Visualize MAE for each cultivar
    plt.figure(figsize=(12, 6))
    
    # 创建条形图 # Create bar chart
    x = range(len(cultivar_metrics_df))
    plt.bar(x, cultivar_metrics_df['MAE'], width=0.6)
    plt.xticks(x, cultivar_metrics_df['Cultivar'], rotation=45)
    plt.title('3dCNN model: Mean absolute error for each apple cultivar (MAE)') #混合模型：各苹果品种的平均绝对误差
    plt.xlabel('Cultivar') #苹果品种
    plt.ylabel('MAE')
    plt.grid(True, axis='y')
    
    # 在每个柱子上添加样本数量 # Add sample count on top of each bar
    for i, row in enumerate(cultivar_metrics_df.itertuples()):
        plt.text(i, row.MAE + 0.02, f'n={row.Count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_3dCNN_cultivar_mae.png', dpi=300)
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
    plt.title('3dCNN model: Actual vs Predicted Values by Cultivar') #混合模型：各品种实际值 vs 预测值 # 
    plt.xlabel('Actual (Brix)') #实际糖度 # 
    plt.ylabel('Predicted (Brix)') #预测糖度
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_3dCNN_cultivar_predictions.png', dpi=300)
    plt.show()
    
    return cultivar_metrics_df

# 加载训练历史并可视化 # Load training history and visualize
def visualize_training_history(history_path, save_dir):
    # 加载训练历史 # Load training history
    history_df = pd.read_csv(history_path)
    
    # 创建图形 # Create figure
    plt.figure(figsize=(16, 6))
    
    # 绘制损失曲线 # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['loss'], label='Training loss')  #训练损失 
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validataion loss') #验证损失
    plt.title('3dCNN model loss curve')  #混合模型损失曲线
    plt.xlabel('Training rounds')  #训练轮次
    plt.ylabel('Mean square error (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 绘制MAE曲线 # Plot MAE curves
    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['mae'], label='Training MAE')
    plt.plot(history_df['epoch'], history_df['val_mae'], label='Validation MAE')
    plt.title('3dCNN model MAE curve') #混合模型MAE曲线
    plt.xlabel('Training rounds') #训练轮次
    plt.ylabel('Mean absolute error (MAE)') #平均绝对误差 
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_3dCNN_training_history.png', dpi=300)
    plt.show()
    
    # 获取最佳验证MAE和对应的轮次 # Get best validation MAE and corresponding epoch
    best_epoch = history_df['val_mae'].idxmin()
    best_val_mae = history_df['val_mae'].min()
    
    print(f" Best validation MAE; 最佳验证MAE: {best_val_mae:.4f} (Rounds; 轮次 {best_epoch})")
    
    return best_epoch, best_val_mae


# 主函数 # Main function
def main():
    print("Start visualistion of hybrix model results; 开始混合模型结果可视化...") 
    
    # 可视化训练历史 # Visualize training history
    print("\n1. Visualize training history; 可视化训练历史")
    best_epoch, best_val_mae = visualize_training_history(history_path, results_dir)
    
    
    # 预测和评估
    print("\n4. Predict and evaluate on the test set; 对测试集进行预测和评估")
    try:
        results_df, mse, rmse, mae, r2 = visualize(predictions, actual_values, results_dir)
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