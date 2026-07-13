from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
import traceback

app = Flask(__name__)

# -----------------------------
# Lazy Load Model
# -----------------------------
model = None

def get_model():
    global model

    if model is None:
        print("========== Loading Model ==========")
        model = load_model("best_highvolume8.keras")
        print("========== Model Loaded Successfully ==========")

    return model


# -----------------------------
# Class Names
# -----------------------------
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


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return "Skin Disease Prediction API is Running!"


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    print("\n==============================")
    print("Prediction Request Received")
    print("==============================")

    try:

        if "image" not in request.files:
            print("ERROR : No image found in request.")
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        print("Uploaded File :", file.filename)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        print("Image Saved :", filepath)

        # -----------------------------
        # Preprocess
        # -----------------------------
        img = image.load_img(filepath, target_size=(300, 300))

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)

        print("Image Preprocessing Completed")

        # -----------------------------
        # Prediction
        # -----------------------------
        model = get_model()

        print("Running Model Prediction...")

        prediction = model.predict(img_array)

        print("Prediction Finished")

        predicted_index = np.argmax(prediction)

        confidence = float(np.max(prediction) * 100)

        print("Predicted Class :", class_names[predicted_index])
        print("Confidence :", confidence)

        os.remove(filepath)

        return jsonify({
            "prediction": class_names[predicted_index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print("\n========== SERVER ERROR ==========")
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)