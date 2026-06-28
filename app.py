from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("best_highvolume8.keras")

# Class names (same order as training)
class_names = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Eczema Photos",
    "Nail Fungus and other Nail Disease",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Warts Molluscum and other Viral Infections"
]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Skin Disease Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    os.remove(filepath)

    return jsonify({
        "prediction": class_names[predicted_index],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)