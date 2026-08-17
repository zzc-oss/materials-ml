# 材料带隙预测系统

基于机器学习的无机材料带隙（band gap）预测工具：输入化学式（如 `Al2O3`），
预测材料带隙（eV）。

## 背景

传统新材料设计依赖大量试错实验，周期长、成本高。
本项目用机器学习方法，从 Materials Project 公开数据库中学习
"成分 → 性能" 的规律，辅助快速筛选候选材料。

## 方法

- **数据**：Materials Project 中的氧化物数据（含 84 种金属元素）
- **特征**：元素属性描述符（电负性、原子半径、离子半径、价电子数、氧占比等）
  + 元素组成 one-hot，由化学式自动生成
- **模型**：随机森林回归（scikit-learn）
- **效果**：测试集 MAE = 0.592 eV，R² = 0.69（仅凭化学式，无需 DFT 计算的密度/体积）

## 技术栈

- Python
- pandas / numpy（数据处理）
- scikit-learn（机器学习）
- pymatgen（化学式解析 + 元素性质）
- mp-api（Materials Project 数据接口）
- Flask（Web 界面）

## 项目结构

```
materials-ml/
├── README.md
├── requirements.txt          # 依赖清单
├── feature_engineering.py    # 化学式 → 特征（共享模块）
├── app.py                    # Flask 网页：输入化学式预测带隙
├── model.joblib              # 训练好的模型
├── model_features.joblib     # 特征名列表（预测时复用）
├── api_key.txt               # Materials Project API 密钥（不提交到 GitHub）
├── data/
│   └── materials.csv         # 下载的氧化物数据
└── notebooks/
    ├── 01_data_exploration.py    # 数据探索
    ├── 02_download_data.py       # 从 Materials Project 下载数据
    └── 03_train_model.py         # 训练模型
```

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2.（可选）下载数据，需在 api_key.txt 里填入 Materials Project 密钥
python notebooks/02_download_data.py

# 3. 训练模型（生成 model.joblib）
python notebooks/03_train_model.py

# 4. 启动网页，浏览器打开 http://127.0.0.1:5000
python app.py
```

## 项目进度

- [x] 项目初始化
- [x] 数据探索（成分-性能关系分析）
- [x] 从 Materials Project 获取真实数据
- [x] 训练性能预测模型（MAE = 0.592 eV）
- [x] Web 界面交互预测（输入完整化学式）
