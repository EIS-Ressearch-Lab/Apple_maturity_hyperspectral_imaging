#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from datetime import datetime, date
from scipy import stats
import re
import glob

# 设置中文字体支持 # Set Chinese font support
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 # Chinese font
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 # Get minus sign

# 设置全局变量 # Set global variables
today = date.today().strftime('%Y-%m-%d')
img_size = 50
# run_id = '2025-03-10run_Hybrid'  # 使用混合模型的run_id

results_path = '/media/2tbdisk3/data/Haidee/Results'  # 结果存储路径

# 查找最新的模型文件 # Find the latest model files
def find_latest_model(model_dir=results_path):
    model_files = glob.glob(f"{model_dir}/*model_file_starch_hybrid.keras")
    if not model_files:
        print(f"Error: No model was found in directory{model_dir}") #print(f"错误: 在 {model_dir} 目录中没有找到模型文件")
        return None
    
    # 按修改时间排序 # Sort by date modified
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model}") #找到最新模型
    return latest_model

# 自动查找最新模型 # Automatically find the latest model
model_path = find_latest_model()

# results_path = 'all_years_results/'
# model_path = f'{results_path}{run_no}_2025-03-10model_trained_hybrid_starch.keras'  # 模型文件路径
# history_path = f'{results_path}{run_no}_2025-03-10history_starch_hybrid.csv'  # 训练历史文件路径 # 


# Get run_no
match = re.search(r'run_\d+', model_path)
run_no = match.group(0) if match else None
print(run_no)

pattern = f'{results_path}/*{run_no}*_history_starch_hybrid.csv'
history_path_matches = glob.glob(pattern)
history_path = history_path_matches[0]


# 创建结果目录
# results_dir = f'hybrid_results_{run_id}_{today}'
results_dir = f'/media/2tbdisk3/data/Haidee/Results/Predictions/{run_no}_hybrid_results'
os.makedirs(results_dir, exist_ok=True)
print(f"Results directory; 结果目录 '{results_dir}' Created successfully; 创建成功!")


# 定义数据生成器函数（与训练文件相同）
def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size):
    num_samples = len(file_list)
    missing_files = []  # 缺失文件列表

    batch_files = file_list
    batch_data = []
    batch_targets = []
    batch_cultivars = []

    # 文件加载和处理
    for i, file in enumerate(batch_files):
        try:
            data = np.load(file)
            batch_data.append(data)
            if targets is not None:
                batch_targets.append(targets[i])
            batch_cultivars.append(cultivars[i])
        except FileNotFoundError:
            missing_files.append(file)
            print(f"File not found; 文件未找到: {file}. Skipping; 跳过...")
            continue

    # 转换为numpy数组
    batch_data = np.array(batch_data)  # 形状: (batch_size, 20, 20, 204)
    if targets is not None:
        batch_targets = np.array(batch_targets)  # 形状: (batch_size,)
    batch_cultivars = np.array(batch_cultivars)  # 形状: (batch_size, 6)
    
    if len(batch_data) == 0:
        return None, None  # 如果没有加载数据则返回None
    
    # 扩展品种信息以匹配输入数据的空间维度 # Expand cultivar information to match the spatial dimensions of the input data
    expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1)
    expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)

    # 在最后一个轴上连接品种信息与原始数据 # Concatenate cultivar information with the original data along the last axis
    combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # 形状: (batch_size, 20, 20, 210)

    return combined_data, batch_targets


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
    plt.title('Hybrid model loss curve')  #混合模型损失曲线
    plt.xlabel('Training rounds')  #训练轮次
    plt.ylabel('Mean square error (MSE)')
    plt.legend()
    plt.grid(True)
    
    # 绘制MAE曲线 # Plot MAE curves
    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['mae'], label='Training MAE')
    plt.plot(history_df['epoch'], history_df['val_mae'], label='Validation MAE')
    plt.title('Hybrid model MAE curve') #混合模型MAE曲线
    plt.xlabel('Training rounds') #训练轮次
    plt.ylabel('Mean absolute error (MAE)') #平均绝对误差 
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/starch_hybrid_training_history.png', dpi=300)
    plt.show()
    
    # 获取最佳验证MAE和对应的轮次 # Get best validation MAE and corresponding epoch
    best_epoch = history_df['val_mae'].idxmin()
    best_val_mae = history_df['val_mae'].min()
    
    print(f" Best validation MAE; 最佳验证MAE: {best_val_mae:.4f} (Rounds; 轮次 {best_epoch})")
    
    return best_epoch, best_val_mae


# 加载测试数据
def load_test_data(test_data_path):
    # 加载测试数据
    X_test_starch = np.load(f'{test_data_path}X_test_all_years_Starch_shuffled.npy')
    Y_test_starch = np.load(f'{test_data_path}Y_test_all_years_Starch_shuffled.npy')
    test_encoder = np.load(f'{test_data_path}X_test_all_years_Starch_encoder_shuffled.npy')
    
    # 添加路径前缀
    spectral_path = '/media/2tbdisk3/data/Haidee/May_2025_spectral_img/Spectral/'
    X_test_starch = [spectral_path + file for file in X_test_starch]
    
    return X_test_starch, Y_test_starch, test_encoder


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
    plt.xlabel('Actual (Starch)') #实际糖度
    plt.ylabel('Predicted (Starch)') # 预测糖度
    plt.grid(True)
    
    # 添加评估指标文本 # Add evaluation metrics text
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/starch_hybrid_test_predictions.png', dpi=300)
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
    plt.savefig(f'{save_dir}/starch_hybrid_error_distribution.png', dpi=300)
    plt.show()
    
    # 保存预测结果和实际值到CSV文件
    results_df = pd.DataFrame({
        'Actual': test_targets,
        'Predicted': predictions,
        'Error': errors
    })
    results_df.to_csv(f'{save_dir}/starch_hybrid_prediction_results.csv', index=False)
    
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
    cultivar_metrics_df.to_csv(f'{save_dir}/starch_hybrid_cultivar_metrics.csv', index=False)
    
    # 打印品种评估指标 # Print cultivar metrics
    print("\n各品种评估指标:")
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
    plt.savefig(f'{save_dir}/starch_hybrid_cultivar_mae.png', dpi=300)
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
    plt.xlabel('Actual (Starch)') #实际糖度 # 
    plt.ylabel('Predicted (Starch)') #预测糖度
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/starch_hybrid_cultivar_predictions.png', dpi=300)
    plt.show()
    
    return cultivar_metrics_df


# 主函数
def main():
    print("Start visualistion of hystarch model results; 开始混合模型结果可视化...") 
    
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