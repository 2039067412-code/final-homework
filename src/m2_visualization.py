import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd

# 设置matplotlib全局字体为支持中文的字体，解决图表中文显示为方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
# 关闭坐标轴负号的Unicode转义，解决负号显示异常的问题
plt.rcParams['axes.unicode_minus'] = False


def run_visualizations(df):
    """
    执行M2模块全部可视化分析，生成4类分析图表并保存至outputs目录
    包含：出行需求时间规律、区域热度排行、车费影响因素、行驶速度热力图
    额外包含可选加分项：基于shapefile的空间分级设色地图

    参数:
        df (pd.DataFrame): 完成清洗与特征工程后的出租车行程数据集
    """
    # ===================== 分析1：出行需求时间规律 =====================
    # 创建画布，设置尺寸为宽10英寸、高6英寸
    plt.figure(figsize=(10, 6))
    # 按「上车小时 + 是否周末」分组，统计每组的订单总数，并重命名计数列为order_count
    hourly_demand = df.groupby(['pickup_hour', 'is_weekend']).size().reset_index(name='order_count')
    # 绘制折线图，x轴为小时，y轴为订单量，按是否周末区分线条颜色，数据点用圆形标记
    sns.lineplot(data=hourly_demand, x='pickup_hour', y='order_count', hue='is_weekend', marker='o')
    # 设置图表标题
    plt.title('工作日与周末分小时订单量对比')
    # 设置x轴标签
    plt.xlabel('小时 (0-23)')
    # 设置y轴标签
    plt.ylabel('总订单量')
    # 设置图例标题，将0/1数值标签替换为可读的中文文本
    plt.legend(title='是否周末', labels=['工作日', '周末'])
    # 设置x轴刻度为0到23的整数，覆盖全天24小时
    plt.xticks(range(0, 24))
    # 显示网格线，设置透明度为0.3避免遮挡数据
    plt.grid(True, alpha=0.3)
    # 保存图片到指定路径，bbox_inches='tight'自动裁剪多余空白区域
    plt.savefig('outputs/m2_1_temporal_demand.png', bbox_inches='tight')
    # 关闭当前画布，释放内存，避免多图叠加
    plt.close()

    # ===================== 分析2：区域热度分析 =====================
    # 创建画布
    plt.figure(figsize=(10, 6))
    # 统计所有上车区域的订单数量，按数量降序排列，取前10名
    top_zones = df['PULocationID'].value_counts().nlargest(10)
    # 绘制横向柱状图，x轴为区域ID（转字符串避免数值排序），y轴为订单量
    # 使用viridis配色，按区域ID区分柱子颜色，关闭图例避免冗余
    sns.barplot(x=top_zones.index.astype(str), y=top_zones.values, hue=top_zones.index.astype(str), palette='viridis', legend=False)
    # 设置图表标题
    plt.title('上客量最高的 TOP 10 区域 (PULocationID)')
    # 设置x轴标签
    plt.xlabel('上客区域ID')
    # 设置y轴标签
    plt.ylabel('订单量')
    # 保存图表
    plt.savefig('outputs/m2_2_top_zones.png', bbox_inches='tight')
    # 关闭画布
    plt.close()

    # ----- 加分项：区域分级设色地图 -----
    # 定义shapefile文件路径
    shp_path = 'data/taxi_zones/taxi_zones.shp'
    # 判断地图文件是否存在，存在则绘制地图，不存在则跳过并提示
    if os.path.exists(shp_path):
        # 读取shapefile地理空间数据，转为GeoDataFrame格式
        gdf = gpd.read_file(shp_path)
        # 统计每个上车区域的订单总量，转为DataFrame格式
        zone_counts = df['PULocationID'].value_counts().reset_index()
        # 重命名列名，方便后续与地理数据按LocationID字段合并
        zone_counts.columns = ['LocationID', 'Demand']
        # 将地理数据与订单需求数据按区域ID左连接，无订单的区域需求量填充为0
        gdf = gdf.merge(zone_counts, on='LocationID', how='left').fillna(0)
        # 创建画布与坐标轴对象，设置画布尺寸
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # 绘制分级设色地图，按Demand字段数值填充颜色
        # 开启图例，使用橙红色系配色方案，设置图例标签文本
        gdf.plot(column='Demand', ax=ax, legend=True, cmap='OrRd', 
                 legend_kwds={'label': "订单量"})
        # 设置地图标题
        plt.title('纽约市出租车各区域出行需求空间分布')
        # 关闭地图坐标轴刻度与边框，保持视觉整洁
        ax.set_axis_off()
        # 保存地图图片
        plt.savefig('outputs/m2_2_map_bonus.png', bbox_inches='tight')
        # 关闭画布
        plt.close()
    else:
        # 文件不存在时，在控制台输出提示信息
        print(f"提示: 未找到地图文件 {shp_path}，已跳过地图绘制。")

    # ===================== 分析3：车费影响因素分析 =====================
    # 创建画布
    plt.figure(figsize=(10, 6))
    # 从全量数据中随机抽样10000条记录，固定随机种子保证结果可复现
    # 抽样目的：避免全量数据散点过度重叠，无法观察趋势，同时提升绘图速度
    sample_df = df.sample(n=10000, random_state=42)
    # 绘制散点图，x轴为行程距离，y轴为车费金额
    # alpha设置点的透明度，s设置点的大小，缓解点重叠问题
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', alpha=0.5, s=10)
    # 设置图表标题，标注抽样数量
    plt.title('行程距离与车费金额关系 (抽样1万条)')
    # 设置x轴标签，附带单位
    plt.xlabel('行程距离 (英里)')
    # 设置y轴标签，附带单位
    plt.ylabel('车费金额 (美元)')
    # 显示网格线
    plt.grid(True, alpha=0.3)
    # 保存图表
    plt.savefig('outputs/m2_3_fare_factors.png', bbox_inches='tight')
    # 关闭画布
    plt.close()

    # ===================== 分析4：自选洞察 - 行驶速度热力图 =====================
    # 创建画布
    plt.figure(figsize=(10, 6))
    # 生成透视表：行索引为星期，列索引为小时，值为对应时段的平均行驶速度
    speed_pivot = df.pivot_table(values='avg_speed_mph', index='pickup_weekday', columns='pickup_hour', aggfunc='mean')
    # 绘制热力图，使用冷暖色调配色，关闭单元格数值标注以保持画面整洁
    sns.heatmap(speed_pivot, cmap='coolwarm', annot=False)
    # 设置图表标题，标注速度单位
    plt.title('周一至周日各小时平均行驶速度热力图 (MPH)')
    # 设置x轴标签
    plt.xlabel('小时 (0-23)')
    # 设置y轴标签，说明星期数值对应关系
    plt.ylabel('星期 (0=周一, 6=周日)')
    # 保存图表
    plt.savefig('outputs/m2_4_custom_speed_heatmap.png', bbox_inches='tight')
    # 关闭画布
    plt.close()
    
    # 全部图表生成完成后，控制台输出完成提示
    print("M2 图表已全部保存至 outputs/ 目录。")