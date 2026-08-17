from flask import Flask, request
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

# ===== HTML 表单（这个是网页的"输入界面"，先照着抄，待会解释）=====
FORM = """
<form method="POST" action="/predict">
    金属元素：
    <select name="metal">
        <option value="Al">Al(铝)</option>
        <option value="Si">Si(硅)</option>
        <option value="Zr">Zr(锆)</option>
        <option value="Ti">Ti(钛)</option>
        <option value="Mg">Mg(镁)</option>
    </select><br><br>
    氧金属比：<input name="ratio" placeholder="比如 2"><br><br>
    密度：<input name="density" placeholder="比如 2.65"><br><br>
    晶胞体积：<input name="volume" placeholder="比如 38"><br><br>
    <button type="submit">预测带隙</button>
</form>
"""

@app.route("/")
def home():
    return FORM

@app.route("/predict", methods=["POST"])
def predict():
    # ===== 填空1：读取用户输入的金属、氧金属比、密度、体积 =====
    metal = request.form["metal"]
    ratio = float(request.form["ratio"])
    density = float(request.form["density"])
    volume = float(request.form["volume"])

    # ===== 填空2：把选的金属转成 5 个 0/1 =====
    # 先全部设 0，再把用户选的那个改成 1
    han = {"Al": 0, "Si": 0, "Zr": 0, "Ti": 0, "Mg": 0}
    han[metal] = 1

    # ===== 填空3：构造特征向量（顺序和训练时一致）=====
    features = [[han["Al"], han["Si"], han["Zr"], han["Ti"], han["Mg"],
                 ratio, density, volume]]

    result = model.predict(features)
    return f"预测带隙: {result[0]:.2f} eV"

if __name__ == "__main__":
    app.run(debug=True)
