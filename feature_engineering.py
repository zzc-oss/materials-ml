# -*- coding: utf-8 -*-
"""
特征工程模块：把化学式变成机器学习特征
======================================
从化学式（如 "Al2O3"）解析出元素组成，再生成两类特征：

  1) 数值特征：元素属性（电负性、原子半径、价电子数…）按配比加权平均
  2) 0/1 特征：是否含某个元素（one-hot）

这样模型能处理任意元素，而不是写死的几种。
训练脚本和网页预测共用本模块，保证特征顺序一致。

注：特征名字符串（如 "电负性"、"氧占比"）故意保留中文——
它们已经作为列名存进了训练好的模型文件，改动会破坏模型，需重训。
"""

import numpy as np
from pymatgen.core import Composition, Element


def parse_composition(formula):
    """把 "Al2O3" 解析成 {元素符号: 摩尔分数}，如 {'Al': 0.4, 'O': 0.6}"""
    comp = Composition(formula)
    frac = comp.fractional_composition.get_el_amt_dict()
    return {str(el): float(v) for el, v in frac.items()}


def _valence_electrons(symbol):
    """从电子排布算最外层价电子数"""
    struct = Element(symbol).full_electronic_structure
    if not struct:
        return 0.0
    highest_shell = max(s[0] for s in struct)
    return float(sum(s[2] for s in struct if s[0] == highest_shell))


def _to_float_or_nan(v):
    """安全转 float：None 或 NaN 统一变成 np.nan，方便加权平均时跳过"""
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _element_properties(symbol):
    """返回单个元素的一堆物理/化学性质"""
    el = Element(symbol)
    return {
        "电负性": _to_float_or_nan(el.X),
        "原子序数": _to_float_or_nan(el.Z),
        "原子质量": _to_float_or_nan(el.atomic_mass),
        "原子半径": _to_float_or_nan(el.atomic_radius),
        "离子半径": _to_float_or_nan(el.average_ionic_radius),
        "周期": _to_float_or_nan(el.row),
        "族": _to_float_or_nan(el.group),
        "价电子数": _valence_electrons(symbol),
        "金属": _to_float_or_nan(el.is_metal),
    }


# 数值特征的固定顺序（训练和预测必须完全一致）
NUMERIC_FEATURES = [
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


def _weighted_average(composition, prop_name):
    """对成分里各元素按摩尔分数加权平均某属性（自动跳过 NaN）"""
    total = 0.0
    weight = 0.0
    for element, fraction in composition.items():
        v = _element_properties(element)[prop_name]
        if v == v:  # 不是 NaN
            total += v * fraction
            weight += fraction
    return total / weight if weight > 0 else 0.0


def numeric_features(formula):
    """返回该化学式对应的数值特征字典"""
    composition = parse_composition(formula)
    return {
        "元素种类数": float(len(composition)),
        "氧占比": composition.get("O", 0.0),
        "电负性": _weighted_average(composition, "电负性"),
        "原子序数": _weighted_average(composition, "原子序数"),
        "原子质量": _weighted_average(composition, "原子质量"),
        "原子半径": _weighted_average(composition, "原子半径"),
        "离子半径": _weighted_average(composition, "离子半径"),
        "周期": _weighted_average(composition, "周期"),
        "族": _weighted_average(composition, "族"),
        "价电子数": _weighted_average(composition, "价电子数"),
        "金属比例": _weighted_average(composition, "金属"),
    }


def collect_elements(formula_list):
    """从训练数据的所有化学式里，收集除 O 之外出现过的元素"""
    elements = set()
    for f in formula_list:
        for e in parse_composition(f):
            if e != "O":
                elements.add(e)
    return sorted(elements)


def generate_feature_names(element_list):
    """特征列名 = 数值特征名 + 每个元素一个「含X」"""
    return NUMERIC_FEATURES + ["含" + e for e in element_list]


def formula_to_features(formula, feature_names):
    """
    把化学式变成按「feature_names」顺序排好的一行特征值。
    训练和网页预测都用它，保证特征顺序一致。
    """
    composition = parse_composition(formula)
    numeric = numeric_features(formula)
    row = []
    for name in feature_names:
        if name.startswith("含"):        # 0/1 特征：是否含某元素
            row.append(1.0 if name[1:] in composition else 0.0)
        else:                            # 数值特征
            row.append(numeric[name])
    return row
