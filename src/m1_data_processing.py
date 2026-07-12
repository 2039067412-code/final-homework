import pandas as pd
import numpy as np
import os


def generate_quality_report(df, output_path):
    """
    生成数据集质量统计报告，并将结果保存为CSV文件
    统计维度包含：字段缺失率、字段唯一值数量、字段数据类型、数值字段的3σ原则异常值比例

    参数:
        df (pd.DataFrame): 待进行质量分析的原始数据集
        output_path (str): 生成的质量报告CSV文件的保存路径
    """
    # 构建质量报告核心数据框，逐列统计三项基础指标
    report = pd.DataFrame({
        '缺失率': df.isnull().mean(),       # 每列空值数量占总行数的比例
        '唯一值数量': df.nunique(),          # 每列中不同取值的总个数
        '数据类型': df.dtypes               # 每列对应的数据类型
    })

    # 初始化列表，用于依次存储每个数值型字段的异常值比例
    outlier_rates = []
    # 遍历所有数值型字段，基于3σ原则计算异常值占比
    # 3σ原则：数值落在 均值±3倍标准差 区间外时，判定为统计意义上的异常值
    for col in df.select_dtypes(include=[np.number]).columns:
        # 计算当前字段的均值与标准差
        mean, std = df[col].mean(), df[col].std()
        # 计算异常值（小于均值-3σ 或 大于均值+3σ）占全部样本的比例
        outliers = ((df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)).mean()
        outlier_rates.append(outliers)

    # 获取全部数值型字段的列名
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 在报告中新增「3Sigma异常值比例」列，初始值全部设为NaN（非数值列保持空值）
    report['3Sigma异常值比例'] = np.nan
    # 仅为数值型字段填充对应的异常值比例计算结果
    report.loc[numeric_cols, '3Sigma异常值比例'] = outlier_rates

    # 将质量报告以UTF-8 BOM编码保存为CSV文件，兼容Excel正常显示中文
    report.to_csv(output_path, encoding='utf-8-sig')
    # 控制台输出保存完成提示
    print(f"数据质量报告已保存至 {output_path}")


def process_data(file_path):
    """
    完整的数据处理流水线：加载原始数据 → 生成数据质量报告 → 执行数据清洗 → 构造特征字段
    包含时间范围过滤、数值异常过滤、基础时间特征提取，以及2个业务衍生特征构造

    参数:
        file_path (str): 原始parquet格式出租车行程数据的文件路径

    返回:
        pd.DataFrame: 完成清洗与特征工程后的最终数据集
    """
    # ========== 步骤1：加载原始数据 ==========
    # 读取parquet格式的出租车行程数据，转为DataFrame格式
    df = pd.read_parquet(file_path)

    # ========== 步骤2：生成原始数据质量报告 ==========
    # 调用质量报告生成函数，输出清洗前的数据质量统计文件
    generate_quality_report(df, 'outputs/data_quality_report.csv')

    # ========== 步骤3：数据清洗策略 ==========
    # 策略1：时间范围过滤
    # 仅保留2026年1月内的上车记录，剔除日期异常的历史脏数据与未来错误数据
    df = df[(df['tpep_pickup_datetime'] >= '2026-01-01') & 
            (df['tpep_pickup_datetime'] < '2026-02-01')]

    # 策略2：行程物理属性过滤
    # 过滤行程距离：剔除距离为0的无效订单，以及超过200英里的极端异常长单
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 200)]
    # 过滤乘客人数：保留1-6人范围的正常订单，剔除0人无效记录与超员异常数据
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

    # 策略3：车费金额过滤
    # 剔除负车费（通常为退款、系统错误）与超过1000美元的天价异常车费
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 1000)]

    # ========== 步骤4：提取基础时间特征 ==========
    # 提取上车时刻的小时数（取值0-23），用于分析日内出行时段规律
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    # 提取上车日期对应的星期数（0=周一，6=周日），用于区分工作日与周末
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    # 标记是否为周末：周六、周日标记为1，工作日标记为0
    df['is_weekend'] = df['pickup_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    # 标记是否为早晚高峰时段
    # 规则：非周末的早高峰（7-9点）、晚高峰（16-19点）判定为高峰，标记为1
    df['is_peak'] = ((~df['is_weekend'].astype(bool)) & 
                     ((df['pickup_hour'].isin([7,8,9])) | (df['pickup_hour'].isin([16,17,18,19])))).astype(int)

    # ========== 步骤5：构造业务衍生特征 ==========
    # 衍生特征1：行程耗时（单位：分钟）
    # 由下车时间减去上车时间换算得到，可用于分析不同区域、时段的拥堵程度与行程效率
    df['trip_duration_mins'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    # 二次清洗：剔除1分钟以内的无效短单，以及5小时以上的极端长单
    df = df[(df['trip_duration_mins'] > 1) & (df['trip_duration_mins'] < 300)]

    # 衍生特征2：平均行驶速度（单位：英里/小时）
    # 由行程距离除以行程时长换算得到，直观反映道路通行效率，可辅助运力调度与拥堵评估
    df['avg_speed_mph'] = df['trip_distance'] / (df['trip_duration_mins'] / 60.0)
    # 二次清洗：剔除超过100英里/小时的不合理超速记录
    df = df[df['avg_speed_mph'] < 100]

    # 控制台输出处理完成提示，以及清洗后的最终数据量
    print(f"数据清洗与特征工程完成，当前数据量: {len(df)}")
    # 返回完成全部处理的数据集
    return df