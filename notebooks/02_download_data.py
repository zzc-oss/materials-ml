# -*- coding: utf-8 -*-
"""
从 Materials Project 下载真实材料数据
======================================
下载氧化物（陶瓷类）材料的性能数据，保存为 CSV，
用于后续的机器学习建模。

运行方式：
    python notebooks/02_download_data.py
"""

import os
from mp_api.client import MPRester

# ============ 1. 读取 API 密钥 ============
# 密钥存在 api_key.txt 里（这个文件不会被提交到 GitHub）
with open("api_key.txt", "r") as f:
    api_key = f.read().strip()

# ============ 2. 定义要下载的材料范围 ============
# 下载所有含氧化合物：二元、三元、四元（含 O + 任意其他元素）

# 要提取的性能字段
fields = [
    "material_id",              # 材料ID
    "formula_pretty",           # 化学式（好看格式）
    "elements",                 # 元素列表
    "formation_energy_per_atom",# 形成能（每原子）
    "energy_above_hull",        # 高于凸包能量（稳定性指标，越低越稳定）
    "band_gap",                 # 带隙（eV）
    "density",                  # 密度（g/cm³）
    "volume",                   # 晶胞体积
]

print("开始下载数据...")
print("查询范围：所有含氧化合物（二元、三元、四元）\n")

all_docs = []
with MPRester(api_key) as mpr:
    for 元数 in [2, 3, 4]:
        print(f"正在查询含 O 的 {元数} 元化合物...")
        docs = mpr.materials.summary.search(
            elements=["O"],
            num_elements=元数,
            fields=fields,
        )
        all_docs.extend(docs)
        print(f"  找到 {len(docs)} 条")

print(f"\n共下载 {len(all_docs)} 条材料数据")

# ============ 3. 转成表格 ============
import pandas as pd

rows = []
for d in all_docs:
    rows.append({
        "material_id": d.material_id,
        "化学式": d.formula_pretty,
        "元素": ",".join(str(e) for e in d.elements),
        "形成能_eV": d.formation_energy_per_atom,
        "稳定性_eV": d.energy_above_hull,
        "带隙_eV": d.band_gap,
        "密度_g_cm3": d.density,
        "晶胞体积": d.volume,
    })

df = pd.DataFrame(rows)

# ============ 4. 清理 + 保存 ============
# 去掉缺失值
df = df.dropna()
print(f"去除缺失值后剩余 {len(df)} 条\n")

# 创建 data 文件夹（如果不存在）
os.makedirs("data", exist_ok=True)
df.to_csv("data/materials.csv", index=False, encoding="utf-8-sig")

print("保存成功！数据预览：")
print(df.head(10))
print(f"\n数据已保存到 data/materials.csv")
