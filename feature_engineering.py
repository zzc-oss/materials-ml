# -*- coding: utf-8 -*-
"""
特征工程模块：把化学式变成机器学习特征
======================================
从化学式（如 "Al2O3"）解析出元素组成，再生成两类特征：

  1) 数值特征：元素属性（电负性、原子半径、价电子数…）按配比加权平均
  2) 0/1 特征：是否含某个元素（one-hot）

这样模型能处理任意元素，而不是写死的几种。
训练脚本和网页预测共用本模块，保证特征顺序一致。
"""

import numpy as np
from pymatgen.core import Composition, Element


def 解析成分(化学式):
    """把 "Al2O3" 解析成 {元素符号: 摩尔分数}，如 {'Al': 0.4, 'O': 0.6}"""
    comp = Composition(化学式)
    frac = comp.fractional_composition.get_el_amt_dict()
    return {str(el): float(v) for el, v in frac.items()}


def _价电子数(元素符号):
    """从电子排布算最外层价电子数"""
    struct = Element(元素符号).full_electronic_structure
    if not struct:
        return 0.0
    最高层 = max(s[0] for s in struct)
    return float(sum(s[2] for s in struct if s[0] == 最高层))


def _数或NaN(v):
    """安全转 float：None 或 NaN 统一变成 np.nan，方便加权平均时跳过"""
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _元素属性(元素符号):
    """返回单个元素的一堆物理/化学性质"""
    el = Element(元素符号)
    return {
        "电负性": _数或NaN(el.X),
        "原子序数": _数或NaN(el.Z),
        "原子质量": _数或NaN(el.atomic_mass),
        "原子半径": _数或NaN(el.atomic_radius),
        "离子半径": _数或NaN(el.average_ionic_radius),
        "周期": _数或NaN(el.row),
        "族": _数或NaN(el.group),
        "价电子数": _价电子数(元素符号),
        "金属": _数或NaN(el.is_metal),
    }


# 数值特征的固定顺序（训练和预测必须完全一致）
数值特征名 = [
    "元素种类数",
    "氧占比",
    "电负性",
    "原子序数",
    "原子质量",
    "原子半径",
    "离子半径",
    "周期",
    "族",
    "价电子数",
    "金属比例",
]


def _加权平均(成分, 属性名):
    """对成分里各元素按摩尔分数加权平均某属性（自动跳过 NaN）"""
    分子 = 0.0
    分母 = 0.0
    for 元素, 分数 in 成分.items():
        v = _元素属性(元素)[属性名]
        if v == v:  # 不是 NaN
            分子 += v * 分数
            分母 += 分数
    return 分子 / 分母 if 分母 > 0 else 0.0


def 数值特征(化学式):
    """返回该化学式对应的数值特征字典"""
    成分 = 解析成分(化学式)
    return {
        "元素种类数": float(len(成分)),
        "氧占比": 成分.get("O", 0.0),
        "电负性": _加权平均(成分, "电负性"),
        "原子序数": _加权平均(成分, "原子序数"),
        "原子质量": _加权平均(成分, "原子质量"),
        "原子半径": _加权平均(成分, "原子半径"),
        "离子半径": _加权平均(成分, "离子半径"),
        "周期": _加权平均(成分, "周期"),
        "族": _加权平均(成分, "族"),
        "价电子数": _加权平均(成分, "价电子数"),
        "金属比例": _加权平均(成分, "金属"),
    }


def 收集元素列表(化学式列表):
    """从训练数据的所有化学式里，收集除 O 之外出现过的元素"""
    els = set()
    for f in 化学式列表:
        for e in 解析成分(f):
            if e != "O":
                els.add(e)
    return sorted(els)


def 生成特征名(元素列表):
    """特征列名 = 数值特征名 + 每个元素一个「含X」"""
    return 数值特征名 + ["含" + e for e in 元素列表]


def 化学式转特征(化学式, 特征名列表):
    """
    把化学式变成按「特征名列表」顺序排好的一行特征值。
    训练和网页预测都用它，保证特征顺序一致。
    """
    成分 = 解析成分(化学式)
    数值 = 数值特征(化学式)
    行 = []
    for name in 特征名列表:
        if name.startswith("含"):        # 0/1 特征：是否含某元素
            行.append(1.0 if name[1:] in 成分 else 0.0)
        else:                            # 数值特征
            行.append(数值[name])
    return 行
