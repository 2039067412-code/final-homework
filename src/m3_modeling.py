import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 固定numpy与PyTorch全局随机种子，保证每次运行的训练过程与结果完全可复现
np.random.seed(42)
torch.manual_seed(42)


class DemandPredictorNN(nn.Module):
    """
    基于全连接神经网络的出行需求量回归预测模型
    采用三层全连接网络结构，以ReLU作为激活函数，最终输出单维度的需求量预测值
    """
    def __init__(self, input_dim):
        # 调用父类nn.Module的初始化方法，完成模块基础初始化
        super(DemandPredictorNN, self).__init__()
        # 第一层全连接层：输入维度为特征数量，输出维度为64维
        self.fc1 = nn.Linear(input_dim, 64)
        # 第一层激活函数：ReLU非线性激活，引入非线性表达能力
        self.relu1 = nn.ReLU()
        # 第二层全连接层：输入64维特征，输出32维特征
        self.fc2 = nn.Linear(64, 32)
        # 第二层激活函数：ReLU非线性激活
        self.relu2 = nn.ReLU()
        # 第三层输出层：将32维特征映射为1维的需求量预测值
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        定义模型前向传播的计算流程
        参数 x: 输入的特征张量
        返回: 模型输出的需求量预测结果
        """
        # 第一层计算：线性变换 + ReLU激活
        x = self.relu1(self.fc1(x))
        # 第二层计算：线性变换 + ReLU激活
        x = self.relu2(self.fc2(x))
        # 输出层：线性变换输出最终预测值，回归任务无需额外激活函数
        x = self.fc3(x)
        return x


def predict_demand(model, location_id, hour, weekday):
    """供 M4 问答系统调用的预测函数"""
    # 根据输入的星期与小时，按规则自动判定是否属于高峰时段
    is_peak = 1 if weekday < 5 and (hour in [7, 8, 9, 16, 17, 18, 19]) else 0
    # 构造与训练时维度一致的特征向量：区域ID、小时、星期、是否高峰
    features = [[location_id, hour, weekday, is_peak]]
    # 调用训练好的模型执行预测，取第一条预测结果
    pred = model.predict(features)[0]
    # 返回取整后的需求量预测值，符合业务场景的整数语义
    return int(pred)


def run_modeling(df):
    # 从上车时间字段中提取日期信息，用于按天维度聚合订单
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    # 按「上车区域+日期+小时+星期+是否高峰」分组，统计每组的订单总数作为需求量
    # 将原始订单级数据聚合为「区域-时段」粒度的样本，构成预测任务的数据集
    agg_df = df.groupby(['PULocationID', 'pickup_date', 'pickup_hour', 'pickup_weekday', 'is_peak']).size().reset_index(name='demand')
    
    # 定义用于需求量预测的输入特征列
    features = ['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_peak']
    # 提取输入特征矩阵 X
    X = agg_df[features].values
    # 提取预测目标标签 y：对应时段对应区域的订单需求量
    y = agg_df['demand'].values

    # 按8:2比例随机划分训练集与测试集，固定随机种子保证划分结果可复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化Z-score标准化器，对特征做标准化处理，适配神经网络的训练要求
    scaler = StandardScaler()
    # 在训练集上拟合标准化参数，并完成训练集特征转换
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用训练集拟合的参数对测试集特征做转换，避免数据泄露
    X_test_scaled = scaler.transform(X_test)

    # ===================== 随机森林模型训练与评估 =====================
    # 初始化随机森林回归器：100棵决策树，固定随机种子，启用全部CPU核心并行训练
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # 在训练集上训练随机森林模型（树模型无需特征标准化，直接使用原始特征）
    rf_model.fit(X_train, y_train)
    # 在测试集上生成预测结果
    rf_preds = rf_model.predict(X_test)
    # 计算随机森林在测试集上的平均绝对误差 MAE
    rf_mae = mean_absolute_error(y_test, rf_preds)
    # 计算随机森林在测试集上的均方根误差 RMSE
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    # ===================== 神经网络模型训练与评估 =====================
    # 将训练集特征与标签转换为PyTorch张量，数据类型为float32
    # 标签通过view调整为二维形状 [样本数, 1]，与网络输出维度对齐
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    # 将测试集特征与标签转换为PyTorch张量
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 实例化神经网络预测模型，传入输入特征的维度
    nn_model = DemandPredictorNN(input_dim=X_train.shape[1])
    # 定义损失函数：均方误差MSE，适用于回归预测任务
    criterion = nn.MSELoss()
    # 定义优化器：Adam自适应优化器，学习率设为0.01，更新网络全部可学习参数
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

    # 设置模型训练的总轮次
    epochs = 100
    # 初始化列表，用于记录每一轮训练的损失值，后续用于绘制loss变化曲线
    losses = []
    
    # 将模型切换为训练模式，启用梯度计算与Dropout等训练专属逻辑
    nn_model.train()
    for epoch in range(epochs):
        # 清空优化器中累积的梯度，避免上一轮梯度与本轮叠加
        optimizer.zero_grad()
        # 前向传播：输入训练特征，得到模型的预测输出
        outputs = nn_model(X_train_t)
        # 计算预测值与真实标签之间的MSE损失
        loss = criterion(outputs, y_train_t)
        # 反向传播：根据损失计算每个参数的梯度
        loss.backward()
        # 参数更新：优化器根据梯度调整网络权重参数
        optimizer.step()
        # 记录当前轮次的损失数值，转为Python原生数值类型
        losses.append(loss.item())

    # 绘制神经网络训练损失变化曲线
    plt.figure()
    # 绘制训练loss折线
    plt.plot(range(epochs), losses, label='Training Loss')
    # 设置图表标题
    plt.title('Neural Network Training Loss')
    # 设置x轴标签
    plt.xlabel('Epoch')
    # 设置y轴标签
    plt.ylabel('MSE Loss')
    # 显示图例
    plt.legend()
    # 将loss曲线图片保存到指定路径
    plt.savefig('outputs/m3_neural_network_loss.png')
    # 关闭当前画布，释放内存资源
    plt.close()

    # 将模型切换为评估模式，关闭Dropout等训练专属操作
    nn_model.eval()
    # 关闭梯度计算，加速推理过程并节省显存占用
    with torch.no_grad():
        # 得到测试集的预测结果，并转换为numpy数组用于后续指标计算
        nn_preds = nn_model(X_test_t).numpy()
    
    # 计算神经网络在测试集上的平均绝对误差 MAE
    nn_mae = mean_absolute_error(y_test, nn_preds)
    # 计算神经网络在测试集上的均方根误差 RMSE
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))

    # 构建两个模型的评估指标对比数据表
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'Neural Network (PyTorch)'],
        'MAE': [rf_mae, nn_mae],
        'RMSE': [rf_rmse, nn_rmse]
    })
    # 将模型评估指标保存为CSV文件，不保留行索引
    metrics_df.to_csv('outputs/m3_model_metrics.csv', index=False)
    

    # 控制台输出训练完成提示
    print("M3 模型训练完毕，指标已保存。")
    # 返回聚合后的数据集与训练好的随机森林模型，供M4问答系统调用
    return agg_df, rf_model
    
    '''
    优劣分析 (见代码注释)：
    1. 随机森林 (RF)：
       - 优势：对非线性、非单调特征（如离散的区域ID、星期）具有天然的包容性，无需做独热编码或复杂标准化，训练速度快且不易过拟合。
       - 劣势：模型体积大，在极大规模数据上预测耗时。
    2. 神经网络 (NN)：
       - 优势：可通过增加网络层数和参数量捕获更复杂的时空交互关系；可通过Embedding层专门处理LocationID。
       - 劣势：对特征缩放敏感，原始类别特征直接输入普通线性层效果不佳；需要精细调参（LR，Batch Size等），否则容易在起步阶段陷入局部最优。
    '''