#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, date
from tensorflow.keras.models import load_model
import re
import glob
import pickle

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置全局变量
today = date.today().strftime('%Y-%m-%d')
img_size = 50
run_no = '24'
batch_size = 8
results_path = '/media/2tbdisk3/data/Haidee/Results'

def find_latest_model(model_dir='/media/2tbdisk3/data/Haidee/Results', run_no=None):
    if run_no is not None:
        model_files = glob.glob(f"{model_dir}/*run_{run_no}*model_trained_2DCNN_brix.keras")
    else:
        model_files = glob.glob(f"{model_dir}/*model_trained_2DCNN_brix.keras")
    if not model_files:
        print(f"Error: No model was found in directory{model_dir}") #print(f"错误: 在 {model_dir} 目录中没有找到模型文件")
        return None
    


    # 按修改时间排序 # Sort by date modified
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model}") #找到最新模型
    return latest_model


# 自动查找最新模型 # Automatically find the latest model
if run_no is None:
    model_path = find_latest_model()
else:
    model_path = find_latest_model(run_no=run_no)

# Get run_no
match = re.search(r'run_\d+', model_path)
run_no = match.group(0) if match else None
print(run_no)

pattern = f'{results_path}/*{run_no}*history_brix.csv'
print(pattern)
history_path_matches = glob.glob(pattern)
history_path = history_path_matches[0]


# 创建结果目录 Create results directory
results_dir = f'/media/2tbdisk3/data/Haidee/Results/Predictions/{run_no}_CNN_results'

os.makedirs(results_dir, exist_ok=True)
print(f"结果目录 '{results_dir}' 创建成功!")


# 定义数据生成器函数（与训练文件相同）
def data_generator_w_cultivar(file_list, targets, cultivars, batch_size, img_size, band_filter = None):
    num_samples = len(file_list)
    # missing_files = []  # 缺失文件列表

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
                    if img_size == 40 or img_size ==20:
                        data_reduced = data[5:-5, 5:-5, :] # Remove 5 pixels from each edge
                        batch_data.append(data_reduced)
                    else:                    
                        batch_data.append(data)
                    batch_targets.append(targets[offset + i])
                    batch_cultivars.append(cultivars[offset + i])
                except FileNotFoundError:

                    continue
            

            # Convert lists to numpy arrays
            batch_data = np.array(batch_data)  # Shape: (batch_size, 50, 50, 204)
            batch_targets = np.array(batch_targets)  # Shape: (batch_size,)
            batch_cultivars = np.array(batch_cultivars)  # Shape: (batch_size, 6)
            
            if len(batch_data) == 0:
                continue # Skip if no data loaded

            # Filter some spectral bands if needed
            if band_filter is not None:

                filtered_test = batch_data[:, :, :, band_filter] # % of the features for brix

                batch_data = filtered_test
            
                       
            # Expand cultivar information to match the input data's spatial dimensions
            expanded_cultivars = np.repeat(batch_cultivars[:, np.newaxis, np.newaxis, :], img_size, axis=1) # Adds singleton dimensions to match the input data's spatial dimensions (batchsize, 1, 1, 1, 6)
            expanded_cultivars = np.repeat(expanded_cultivars, img_size, axis=2)

            # Concatenate cultivar information with the original data along the last axis
            combined_data = np.concatenate([batch_data, expanded_cultivars], axis=-1)  # Shape: (batch_size, 50, 50, 210)


            # Yield the combined data and targets
            yield combined_data, batch_targets


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


# 对测试集进行预测并可视化
def predict_and_visualize(model, X_test, Y_test, test_encoder, img_size, save_dir):
    print("Preparing test data...")
    test_generator = data_generator_w_cultivar(X_test, Y_test, test_encoder, batch_size, img_size, band_filter=None)

    print("Make predictions; 进行预测...")
    predictions = []
    targets = []

    for x_batch, y_batch in test_generator:
        preds = model.predict(x_batch)
        predictions.extend(preds.flatten())
        targets.extend(y_batch)
        if len(targets) >= len(X_test):  # Stop after full test set
            break

    predictions = np.array(predictions[:len(X_test)])
    test_targets = np.array(targets[:len(X_test)])
    
    # 计算评估指标
    mse, rmse, mae, r2 = calculate_metrics(test_targets, predictions)
    
    # 打印评估结果 # Print evaluation results
    print(f"\nTest set evaluation results:") #测试集评估结果
    print(f"Mean Square Error (MSE): {mse:.4f}") #均方误差
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}") #均方根误差
    print(f"Mean Absolute Error (MAE): {mae:.4f}") #
    print(f"Coefficient of determination (R²): {r2:.4f}") #决定系数
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(test_targets, predictions, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.title('Actual vs Predicted Values') #实际值 vs 预测值
    plt.xlabel('Actual Brix')
    plt.ylabel('Predicted Brix')
    plt.grid(True)
    
    # 添加评估指标文本
    plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_test_predictions.png', dpi=300)
    plt.show()
    
    # 创建误差分布直方图
    errors = predictions - test_targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, density=True)
    
    # 添加核密度估计曲线
    from scipy import stats
    x = np.linspace(min(errors), max(errors), 100)
    kde = stats.gaussian_kde(errors)
    plt.plot(x, kde(x), 'r-', linewidth=2)
    
    plt.title('Predicted error distribution')
    plt.xlabel('Predicted Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(x=0, color='r', linestyle='--')
    
    # 添加误差统计信息
    plt.text(0.05, 0.95, f'Mean Error: {np.mean(errors):.4f}\nStandard Deviation: {np.std(errors):.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_error_distribution.png', dpi=300)
    plt.show()
    
    # 保存预测结果和实际值到CSV文件
    results_df = pd.DataFrame({
        'Actual': test_targets,
        'Predicted': predictions,
        'Error': errors
    })
    results_df.to_csv(f'{save_dir}/brix_prediction_results.csv', index=False)
    
    return results_df, mse, rmse, mae, r2


# 可视化不同品种的预测性能
def visualize_by_cultivar(results_df, test_encoder, save_dir):
    # 创建品种名称列表
    cultivar_names = ['Braeburn', 'Cox', 'Fuji', 'Gala', 'Golden Delicious', 'Jazz', 'Other']
    
    # 获取每个样本的品种（取每个独热编码的最大索引）
    cultivar_indices = np.argmax(test_encoder, axis=1)
    
    # 将品种索引添加到结果DataFrame
    results_df['Cultivar_Index'] = cultivar_indices
    results_df['Cultivar'] = [cultivar_names[idx] for idx in cultivar_indices]
    
    # 按品种分组计算评估指标
    cultivar_metrics = []
    for i, name in enumerate(cultivar_names):
        cultivar_data = results_df[results_df['Cultivar_Index'] == i]
        if len(cultivar_data) > 0:  # 确保有该品种的数据
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
    
    # 创建品种评估指标DataFrame
    cultivar_metrics_df = pd.DataFrame(cultivar_metrics)
    cultivar_metrics_df.to_csv(f'{save_dir}/brix_cultivar_metrics.csv', index=False)
    
    # 打印品种评估指标
    print("\nEvaluation indicators for each cultivar; 各品种评估指标:")
    print(cultivar_metrics_df)
    
    # 可视化各品种的MAE
    plt.figure(figsize=(12, 6))
    
    # 创建条形图
    x = range(len(cultivar_metrics_df))
    plt.bar(x, cultivar_metrics_df['MAE'], width=0.6)
    plt.xticks(x, cultivar_metrics_df['Cultivar'], rotation=45)
    plt.title('Average absolute error for each cultivar (MAE)') #各苹果品种的平均绝对误差 #
    plt.xlabel('Cultivar') #苹果品种
    plt.ylabel('MAE')
    plt.grid(True, axis='y')
    
    # 在每个柱子上添加样本数量
    for i, row in enumerate(cultivar_metrics_df.itertuples()):
        plt.text(i, row.MAE + 0.02, f'n={row.Count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_cultivar_mae.png', dpi=300)
    plt.show()
    
    # 创建各品种的散点图
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(cultivar_names):
        cultivar_data = results_df[results_df['Cultivar_Index'] == i]
        if len(cultivar_data) > 0:  # 确保有该品种的数据
            plt.scatter(cultivar_data['Actual'], cultivar_data['Predicted'], 
                       label=f'{name} (n={len(cultivar_data)})', alpha=0.7)
    
    plt.plot([min(results_df['Actual']), max(results_df['Actual'])], 
             [min(results_df['Actual']), max(results_df['Actual'])], 'k--')
    plt.title('Actual vs Predicted Values by Cultivar')
    plt.xlabel('Actual Brix')
    plt.ylabel('Predicted Brix')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/brix_cultivar_predictions.png', dpi=300)
    plt.show()
    
    return cultivar_metrics_df


# 主函数 Main function
def main():
    print("Start visualistion of vit model results...")
    
    # 加载模型
    print(f"\n1. Loaded model: {model_path}")
    try:
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test data
    print("\n2. Load test data")
    if img_size == 30 or img_size == 20:
        training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/30px/all_years/'
    elif img_size == 50 or img_size ==40:
        training_data_path = '/home/ht21074/Auto_box_apples/Auto_box_apples/Source_folder/Training_data_May2025/50px/all_years/'

    try:
        X_test, Y_test, test_encoder = load_test_data(training_data_path)
        print(f"测试样本数量: {len(X_test)}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # 预测和评估
    print("\n3. Predict and evaluate on the test set")
    try:
        results_df, mse, rmse, mae, r2 = predict_and_visualize(model, X_test, Y_test, test_encoder, img_size, results_dir)
    except Exception as e:
        print(f"Error during prediction and evaluation: {e}")
        return
    
    # 按品种可视化
    print("\n4. Visualize prediction performance by cultivar")
    try:
        cultivar_metrics_df = visualize_by_cultivar(results_df, test_encoder, results_dir)
    except Exception as e:
        print(f"Error analyzing by cultivar: {e}")
        return

    print(f"\nAll results have been saved to '{results_dir}' directory")


if __name__ == "__main__":
    main() 