# -*- coding: utf-8 -*-
"""
训练随机森林模型预测材料带隙（进阶版）
======================================
输入：化学式 → 元素属性描述符 + 元素组成 one-hot
输出：预测材料带隙（eV）

相比旧版（只含 Al/Si/Zr/Ti/Mg 5 个元素 + 密度/体积）：
  1. 自动识别数据里所有元素，不再写死
  2. 用元素性质（电负性、原子半径、价电子…）做特征
  3. 只用化学式就能预测，网页端直接输入化学式

运行方式（在项目根目录 materials-ml 下）：
    python notebooks/03_train_model.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 让项目根目录可被 import（因为脚本放在 notebooks/ 子目录里）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_engineering as fe

# ============ 1. 读数据 ============
df = pd.read_csv("data/materials.csv")

# ============ 2. 收集所有元素、生成特征列 ============
元素列表 = fe.收集元素列表(df["化学式"])
特征名列表 = fe.生成特征名(元素列表)
print(f"数据 {len(df)} 条，识别到 {len(元素列表)} 种非氧元素")
print(f"特征总数：{len(特征名列表)}"
      f"（数值 {len(fe.数值特征名)} + 元素 {len(元素列表)}）\n")

# ============ 3. 把所有化学式转成特征矩阵 ============
X = pd.DataFrame(
    df["化学式"].apply(lambda f: fe.化学式转特征(f, 特征名列表)).tolist(),
    columns=特征名列表,
)
y = df["带隙_eV"]

# ============ 4. 划分训练集测试集 ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============ 5. 训练 ============
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42)
model.fit(X_train, y_train)

# ============ 6. 评估 ============
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("平均绝对误差 MAE =", round(mae, 3), "eV")
print("决定系数 R2 =", round(r2, 3))

# ============ 7. 特征重要性 ============
importance = pd.Series(model.feature_importances_, index=特征名列表)
importance = importance.sort_values(ascending=False)
print("\n最重要的 15 个特征：")
for name, value in importance.head(15).items():
    print(f"  {name}: {value:.3f}")

# ============ 8. 保存模型 + 特征信息 ============
joblib.dump(model, "model.joblib", compress=3)
joblib.dump({"特征名列表": 特征名列表, "元素列表": 元素列表}, "model_features.joblib")
print("\n模型已保存到 model.joblib，特征信息保存到 model_features.joblib")
