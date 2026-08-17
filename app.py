# -*- coding: utf-8 -*-
from flask import Flask, request

import joblib
import pandas as pd

import feature_engineering as fe

app = Flask(__name__)
model = joblib.load("model.joblib")
特征名列表 = joblib.load("model_features.joblib")["特征名列表"]

# ===== HTML 表单：输入完整化学式 =====
FORM = """
<h2>材料带隙预测（进阶版）</h2>
<p>输入任意氧化物化学式，例如 Al2O3、MgO、TiO2</p>
<form method="POST" action="/predict">
    化学式：<input name="formula" placeholder="比如 Al2O3" autofocus><br><br>
    <button type="submit">预测带隙</button>
</form>
"""


@app.route("/")
def home():
    return FORM


@app.route("/predict", methods=["POST"])
def predict():
    formula = request.form["formula"].strip()
    try:
        features = pd.DataFrame(
            [fe.化学式转特征(formula, 特征名列表)], columns=特征名列表
        )
        result = model.predict(features)
        return (
            f"<p>化学式 <b>{formula}</b> 预测带隙："
            f"<b>{result[0]:.2f} eV</b></p>"
            f"<p><a href='/'>返回</a></p>"
        )
    except Exception as e:
        return (
            f"<p>解析失败：{e}</p>"
            f"<p>请检查化学式格式（如 Al2O3）</p>"
            f"<p><a href='/'>返回</a></p>"
        )


if __name__ == "__main__":
    app.run(debug=True)
