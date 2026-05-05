from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GRADCAM_FOLDER"] = GRADCAM_FOLDER

# ===== Load Model =====
model = tf.keras.models.load_model("model.keras", compile=False)

# Initialize the model to avoid sequential errors
_ = model(tf.random.normal((1, 224, 224, 3)))

# ===== Class Names =====
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ===== Preprocess =====
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ===== Prediction =====
def predict(img_path):
    img = preprocess(img_path)
    pred = model.predict(img)
    idx = np.argmax(pred)
    return class_names[idx], float(np.max(pred)), img

# ===== Grad-CAM (FINAL STABLE VERSION) =====

# ===== ROUTE =====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        filename = secure_filename(file.filename)

        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(img_path)

        label, conf, img_array = predict(img_path)

        grad_path = os.path.join(app.config["GRADCAM_FOLDER"], filename)
       

        return render_template(
            "result.html",
            label=label,
            confidence=round(conf * 100, 2),
            image=img_path,
            
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)