# -*- coding: utf-8 -*-
"""
材料带隙预测 Web 界面（Gradio 版）
==================================
输入化学式，预测材料带隙（eV）。
部署到 Hugging Face Spaces 后，任何人通过浏览器即可使用。
"""

import joblib
import pandas as pd
import gradio as gr

import feature_engineering as fe

# 加载模型和特征名
model = joblib.load("model.joblib")
feature_names = joblib.load("model_features.joblib")["feature_names"]


def predict_band_gap(formula):
    """输入化学式，返回预测带隙的文本"""
    formula = (formula or "").strip()
    if not formula:
        return "请输入化学式"
    try:
        features = pd.DataFrame(
            [fe.formula_to_features(formula, feature_names)], columns=feature_names
        )
        result = model.predict(features)
        return f"{result[0]:.2f} eV"
    except Exception as e:
        return f"解析失败：{e}\n\n请检查化学式格式（如 Al2O3）"


demo = gr.Interface(
    fn=predict_band_gap,
    inputs=gr.Textbox(
        label="化学式",
        placeholder="例如 Al2O3、MgO、TiO2",
    ),
    outputs=gr.Textbox(label="预测带隙"),
    title="材料带隙预测",
    description=(
        "输入氧化物化学式，预测材料带隙（eV）。"
        "基于 Materials Project 数据训练，测试集 MAE = 0.592 eV。"
    ),
    examples=["Al2O3", "MgO", "TiO2", "SiO2", "CaO"],
)

demo.launch()
