import re
import os
import sys
from pathlib import Path
import pandas as pd
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.m3_modeling import predict_demand
except ModuleNotFoundError:
    from m3_modeling import predict_demand

def run_qa_system(df, model_data):
    agg_df, rf_model = model_data
    
    # 加载区域映射字典
    zone_mapping = {}
    lookup_path = 'data/taxi_zone_lookup.csv'
    if os.path.exists(lookup_path):
        lookup_df = pd.read_csv(lookup_path)
        zone_mapping = dict(zip(lookup_df['LocationID'], lookup_df['Zone']))

    def qa_bot(query, history):
        if not query:
            return "请输入您的问题"
            
        # 1. 时段查询 (模糊匹配：高峰)
        if re.search(r'高峰', query):
            peak_orders = df[df['is_peak'] == 1].shape[0]
            return f"**[结论]**: {peak_orders} 单\n\n**[解释]**: 2026年1月数据中，标记为高峰期（工作日7-9点、16-19点）的总订单量为{peak_orders}。\n\n**[文件路径]**: outputs/m2_1_temporal_demand.png\n\n![趋势图](file/outputs/m2_1_temporal_demand.png)"
            
        # 2. 区域排名 (模糊匹配：排名/热门/最多，并动态提取数字)
        elif re.search(r'排名|热门|最多', query) and re.search(r'区域|地点|上客', query):
            # 动态提取用户问的是前几名，找不到数字则默认是 5
            num_match = re.search(r'(\d+)', query)
            top_n_num = int(num_match.group(1)) if num_match else 5
            # 防止用户输入过大的数字导致显示错乱
            top_n_num = min(top_n_num, 50) 
            
            top_n_data = df['PULocationID'].value_counts().head(top_n_num)
            top_names = [zone_mapping.get(zone_id, str(zone_id)) for zone_id in top_n_data.index]
            
            table_str = "| 排名 | 区域ID | 区域名称 | 订单数 |\n|---|---|---|---|\n"
            for rank, (idx, val) in enumerate(top_n_data.items(), 1):
                name = zone_mapping.get(idx, str(idx))
                table_str += f"| {rank} | {idx} | {name} | {val} |\n"
            
            return f"**[结论]**: 排名前 {top_n_num} 的区域为 {top_names}\n\n**[解释]**: 上客量最高的 {top_n_num} 个区域及其订单数分别为：\n\n{table_str}\n\n**[文件路径]**: outputs/m2_2_top_zones.png\n\n![热门区域](file/outputs/m2_2_top_zones.png)"

        # 3. 需求预测 (动态提取星期、小时、区域)
        elif re.search(r'预测|需求|多少单', query) and re.search(r'区域', query):
            day_match = re.search(r'周(一|二|三|四|五|六|日)', query)
            hour_match = re.search(r'(\d+)[点时]', query)
            zone_match = re.search(r'区域(\d+)', query)
            
            if day_match and hour_match and zone_match:
                day_map = {'一':0, '二':1, '三':2, '四':3, '五':4, '六':5, '日':6}
                weekday_str = day_match.group(1)
                weekday = day_map[weekday_str]
                hour = int(hour_match.group(1))
                zone_id = int(zone_match.group(1))
                
                pred_demand = predict_demand(rf_model, zone_id, hour, weekday)
                return f"**[结论]**: 预测需求量约为 {pred_demand} 单/小时\n\n**[解释]**: 已经为您调用 M3 的随机森林预测模型。在周{weekday_str}的 {hour} 点，区域 {zone_id} ({zone_mapping.get(zone_id, '未知区域')}) 的预测打车需求为 {pred_demand} 单。\n\n**[文件路径]**: outputs/m3_model_metrics.csv"
            else:
                return "触发了【需求预测】功能，但您提供的信息不全。请包含完整的预测条件，例如：'预测周一10点区域237的需求'"
                
        # 4. 费用估算 (模糊匹配：车费/钱/金额)
        elif re.search(r'车费|多少钱|金额', query):
            avg_fare = df['fare_amount'].mean()
            return f"**[结论]**: {avg_fare:.2f} 美元\n\n**[解释]**: 清洗后的历史数据中，平均每单的车费金额为 {avg_fare:.2f} 美元。\n\n**[文件路径]**: outputs/m2_3_fare_factors.png\n\n![车费散点图](file/outputs/m2_3_fare_factors.png)"
            
        # 5. 趋势图/图表检索 (模糊匹配：趋势/图表/折线)
        elif re.search(r'趋势|图表|折线|走势', query):
            weekend_orders = df[df['is_weekend'] == 1].shape[0]
            return f"**[结论]**: 周末总订单量为 {weekend_orders} 单\n\n**[解释]**: 系统已提取时间趋势图表，该图表直观展示了工作日与周末分小时的需求对比。\n\n**[文件路径]**: outputs/m2_1_temporal_demand.png\n\n![趋势图](file/outputs/m2_1_temporal_demand.png)"

        # 6. 模型训练收敛情况 (提取 M3 的 Loss 曲线)
        elif re.search(r'收敛|loss|损失|训练', query, re.IGNORECASE):
            return "**[结论]**: 神经网络模型经过 100 个 Epoch 的训练，Loss 值已趋于稳定。\n\n**[解释]**: 下图展示了 PyTorch 神经网络在训练过程中的 MSE Loss 下降曲线，反映了模型对时空特征的拟合过程。\n\n**[文件路径]**: outputs/m3_neural_network_loss.png\n\n![Loss曲线](file/outputs/m3_neural_network_loss.png)"

        # 7. 空间分布地图 (提取 M2 的加分项地图)
        elif re.search(r'地图|空间分布|分布图', query):
            return "**[结论]**: 出行需求高度集中在曼哈顿及两大机场区域。\n\n**[解释]**: 下图基于 Geopandas 绘制，将各个 Taxi Zone 的历史需求量映射到了纽约市的真实地理空间中，颜色越深代表订单密度越大。\n\n**[文件路径]**: outputs/m2_2_map_bonus.png\n\n![空间分布地图](file/outputs/m2_2_map_bonus.png)"

        # 8. 行驶速度与拥堵分析 (提取 M2 的自选洞察热力图)
        elif re.search(r'速度|堵车|拥堵', query):
            return "**[结论]**: 工作日早晚高峰时段平均行驶速度明显下降，存在显著的交通拥堵。\n\n**[解释]**: 下图是通过行程距离与耗时计算出的“一周各小时平均行驶速度热力图”。蓝色区域代表车速较低（拥堵），红色代表车速较高（畅通）。\n\n**[文件路径]**: outputs/m2_4_custom_speed_heatmap.png\n\n![速度热力图](file/outputs/m2_4_custom_speed_heatmap.png)"

        else:
            return "系统暂未识别该意图，请尝试更换关键词，如“车费大概多少”、“排名前3的区域”、“查看趋势图”、“早晚高峰期”、“预测周三8点区域132的需求”等。"

    print("\n[INFO] 正在启动网页交互界面...")
    
    demo = gr.ChatInterface(
        fn=qa_bot,
        title="🚕 城市出租车出行数据智能问答系统",
        description="在下方输入框中提问，系统将自动调用 M1-M3 的结果进行解答并展示图表。",
        examples=[
            "高峰期订单量是多少？",
            "排名前1的区域有哪些？",
            "预测一下周五18点区域161的打车需求",
            "纽约打车平均车费是多少钱？",
            "生成订单量的时间走势图"
        ]
    )
    
    demo.launch(allowed_paths=["outputs"], inbrowser=True)