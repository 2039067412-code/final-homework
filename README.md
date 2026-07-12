# 🚕 城市出租车出行数据分析与智能问答系统

## 📌 项目简介

本项目为《人工智能编程语言》期末大作业 1。项目基于 2026 年 1 月纽约市黄色出租车（Yellow Taxi）的公开行程数据，构建了一个完整的数据科学与人工智能流水线。项目集成了数据清洗、可视化分析、机器学习/深度学习需求预测（随机森林与 PyTorch 神经网络），并最终通过 Gradio 封装了一个支持自然语言交互与动态参数提取的 Web 智能问答系统。

\---

## 📁 目录架构说明

在运行本项目前，请确保您的项目文件树严格遵循以下结构：
final\_project/
├── data/                                 # 存放所有原始数据
│   ├── yellow\_tripdata\_2026-01.parquet   # 2026年1月黄色出租车行程主数据
│   ├── taxi\_zone\_lookup.csv              # 区域ID与名称映射表 (用于M4智能地名解析)
│   └── taxi\_zones/                       # 地理空间 shapefile 文件包 (用于M2地图绘制)
│       ├── taxi\_zones.shp
│       ├── taxi\_zones.shx
│       ├── taxi\_zones.dbf
│       ├── taxi\_zones.prj
│       ├── taxi\_zones.cpg
│       ├── taxi\_zones\_2.shp
│       └── taxi\_zones\_2.cpg
├── outputs/                              # 脚本运行后自动生成的图表与报告目录
├── src/                                  # 核心功能代码包
│   ├── \_\_init\_\_.py                       # 模块声明文件 (空文件)
│   ├── m1\_data\_processing.py             # M1：数据处理与特征工程
│   ├── m2\_visualization.py               # M2：数据分析与可视化
│   ├── m3\_modeling.py                    # M3：PyTorch与随机森林预测建模
│   └── m4\_qa\_system.py                   # M4：基于 Gradio 的智能问答 Web UI
├── main.py                               # 项目全局唯一主入口
├── requirements.txt                      # 项目环境依赖清单
└── 人机协作报告.md                       # AI 辅助编程交互记录与反思总结




## ⚙️ 环境依赖与安装

1. **Python 版本要求**：建议使用 `Python 3.9` 或更高版本。
2. **安装依赖**：在终端（Terminal）中切换至项目根目录 `final\_project/`，执行以下命令安装所有必需的第三方库：

```bash
pip install -r requirements.txt
```

> 注：`requirements.txt` 中已显式包含 `pyogrio` 引擎，以确保 `geopandas` 在各类系统中均能稳定读取底层的 Shapefile 空间数据。



## 🗄️ 数据准备

在运行程序前，请前往 NYC TLC 官网下载以下必要文件，并放置在对应的 `data/` 目录下：

1. **Yellow Taxi Trip Records (PARQUET)**: 2026 年 1 月的数据。
2. **Taxi Zone Lookup Table (CSV)**: 用于 M4 问答系统的区域名称查询。
3. **Taxi Zone Shapefile**: 下载后解压，将 `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` 等所有地理空间关联文件统一放入 `data/taxi\_zones/` 文件夹中。



## 🚀 运行操作指南

**本项目采用高度解耦的模块化设计，请务必从主入口启动。请勿单独运行 `src/` 目录下的子脚本。**

1. 打开终端（VS Code Terminal 或系统命令行），确保当前工作目录位于 `final\_project/`。
2. 输入以下命令启动项目：

```bash
python main.py
```

3. **运行流程与预期结果**：

   * **M1 阶段**：系统将自动读取 parquet 文件，过滤异常值，执行特征工程，并在控制台打印清洗后的数据量。同时在 `outputs/` 目录生成 `data\_quality\_report.csv`。
   * **M2 阶段**：系统将进行分组统计与空间映射，静默生成多张数据分析图表（折线图、条形图、热力图及区域分级设色地图），并保存至 `outputs/` 目录。
   * **M3 阶段**：系统将拆分数据集并执行随机森林与 PyTorch 神经网络的训练。训练完成后，会在 `outputs/` 目录保存网络 Loss 曲线图 `m3\_neural\_network\_loss.png` 以及模型性能评估指标 `m3\_model\_metrics.csv`。
   * **M4 阶段**：控制台将输出 `\\\[INFO] 正在启动网页交互界面...` 以及一条 Local URL（例如 `\\\[http://127.0.0.1:7860](http://127.0.0.1:7860)`）。
4. 系统会自动在默认浏览器中打开问答界面。您可以在聊天框中点击系统提供的示例问题，或者输入自然语言（例如 *"预测一下周五18点区域161的打车需求"*），系统将动态提取参数、调用预测模型，并图文并茂地返回结果。



## 🧩 核心功能模块说明

* **M1 数据处理**：制定了严格的清洗策略（时间越界、距离异常、天价车费剔除等），并基于时间戳提取了 `pickup\_hour`、`pickup\_weekday`，以及高级衍生特征 `is\_peak`（动态高峰期）、`trip\_duration\_mins`（耗时）和 `avg\_speed\_mph`（平均速度）。
* **M2 分析可视化**：涵盖时间规律、区域热度空间分布（Geopandas 地图渲染）、车费影响因素散点图，以及深度的平均行驶速度热力图洞察。
* **M3 预测模型**：将时空离散特征标准化后送入两层 ReLU 隐藏层的全连接网络中，与基于决策树构建的 RandomForest 回归器进行效果与特性对比。
* **M4 智能问答接口**：使用强大的正则匹配引擎动态捕获用户自然语言中的业务实体（时间、日期、区域数字），将变量传递给 M3 的机器学习模型进行实时推理，并通过 Gradio Chat 界面实现 Markdown 友好的多模态响应。

