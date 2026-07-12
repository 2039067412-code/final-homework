import os
import pandas as pd
from src.m1_data_processing import process_data
from src.m2_visualization import run_visualizations
from src.m3_modeling import run_modeling
from src.m4_qa_system import run_qa_system

def main():
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    data_path = 'data/yellow_tripdata_2026-01.parquet'
    
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}。请先下载并放入 data 目录。")
        return

    print("=== 开始执行 M1: 数据处理 ===")
    df_clean = process_data(data_path)
    
    print("\n=== 开始执行 M2: 分析可视化 ===")
    run_visualizations(df_clean)
    
    print("\n=== 开始执行 M3: 预测模型 ===")
    model_data = run_modeling(df_clean)
    
    print("\n=== 开始执行 M4: 问答接口 ===")
    run_qa_system(df_clean, model_data)

if __name__ == "__main__":
    main()