# -*- coding: utf-8 -*-
"""
材料性能数据探索
==============
第一步：了解陶瓷配方数据长什么样，分析成分、工艺与性能之间的关系。

运行方式：
    python notebooks/01_data_exploration.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，避免中文显示成方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 准备一份示例数据 ============
# 这是模拟的陶瓷配方数据，后续会替换成 Materials Project 的真实数据
data = {
    "配方编号": ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"],
    "氧化铝含量_%": [25, 30, 35, 40, 45, 50, 55, 60],
    "烧结温度_℃": [1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650],
    "保温时间_h": [2, 2, 3, 3, 4, 4, 5, 5],
    "抗弯强度_MPa": [180, 210, 260, 250, 230, 220, 190, 170],
}

df = pd.DataFrame(data)

# ============ 2. 查看数据 ============
print("=" * 50)
print("数据概览：")
print("=" * 50)
print(f"共 {df.shape[0]} 条数据，{df.shape[1]} 个变量\n")
print(df, "\n")

print("统计信息：")
print(df.describe().round(2), "\n")

# ============ 3. 相关性分析 ============
print("=" * 50)
print("各因素与抗弯强度的相关系数：")
print("=" * 50)
# 只对数字列算相关性（排除"配方编号"这种文字列）
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()["抗弯强度_MPa"].sort_values(ascending=False)
for col, value in corr.items():
    if col != "抗弯强度_MPa":
        print(f"  {col}: {value:.3f}")
print("\n说明：越接近 1 表示正相关越强，越接近 -1 表示负相关越强\n")

# ============ 4. 画图：看趋势 ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1：氧化铝含量 vs 强度
axes[0].plot(df["氧化铝含量_%"], df["抗弯强度_MPa"], marker="o", color="#2196F3")
axes[0].set_xlabel("氧化铝含量 (%)")
axes[0].set_ylabel("抗弯强度 (MPa)")
axes[0].set_title("氧化铝含量 vs 强度")
axes[0].grid(True, alpha=0.3)

# 图2：烧结温度 vs 强度
axes[1].plot(df["烧结温度_℃"], df["抗弯强度_MPa"], marker="s", color="#FF9800")
axes[1].set_xlabel("烧结温度 (℃)")
axes[1].set_ylabel("抗弯强度 (MPa)")
axes[1].set_title("烧结温度 vs 强度")
axes[1].grid(True, alpha=0.3)

# 图3：保温时间 vs 强度
axes[2].scatter(df["保温时间_h"], df["抗弯强度_MPa"], s=80, color="#4CAF50")
axes[2].set_xlabel("保温时间 (h)")
axes[2].set_ylabel("抗弯强度 (MPa)")
axes[2].set_title("保温时间 vs 强度")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/exploration_result.png", dpi=150)
plt.show()

# ============ 5. 找出最优配方 ============
print("=" * 50)
print("强度最高的前 3 个配方：")
print("=" * 50)
top3 = df.sort_values("抗弯强度_MPa", ascending=False).head(3)
print(top3, "\n")

print("分析完成！图表已保存为 notebooks/exploration_result.png")
