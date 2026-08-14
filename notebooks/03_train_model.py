# -*- coding: utf-8 -*-
"""
训练随机森林模型预测材料带隙
==============================
输入：元素组成（0/1 特征）+ 密度 + 晶胞体积
输出：预测材料带隙（eV）

运行方式（在项目根目录 materials-ml 下）：
    python notebooks/03_train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ============ 1. 读数据 ============
df = pd.read_csv("data/materials.csv")

# ============ 2. 特征工程：把"元素"文字变成 0/1 数字 ============
df["含Al"] = df["元素"].str.contains("Al").astype(int)
df["含Si"] = df["元素"].str.contains("Si").astype(int)
df["含Zr"] = df["元素"].str.contains("Zr").astype(int)
df["含Ti"] = df["元素"].str.contains("Ti").astype(int)
df["含Mg"] = df["元素"].str.contains("Mg").astype(int)

# ============ 3. 确定输入 X 和目标 y ============
feature_cols = ["含Al", "含Si", "含Zr", "含Ti", "含Mg", "密度_g_cm3", "晶胞体积"]
X = df[feature_cols]
y = df["带隙_eV"]

# ============ 4. 划分训练集和测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============ 5. 训练模型 ============
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============ 6. 预测并评估 ============
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("平均绝对误差 MAE =", round(mae, 3), "eV")

# ============ 7. 看哪些特征最重要 ============
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)
print("\n特征重要性排名：")
for name, value in importance.items():
    print(f"  {name}: {value:.3f}")
