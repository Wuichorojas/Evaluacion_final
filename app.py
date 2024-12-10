from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model("crop_classifier_model.h5")

# Etiquetas de las clases
class_labels = ['Almendra', 'Plátano', 'Cereza', 'Arroz', 'Limón', 'Tomate']

# Preprocesar la imagen
def preprocess_image(img, img_height=150, img_width=150):
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Ruta principal para la página web
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para predecir desde un archivo subido
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se subió ningún archivo"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # Guardar temporalmente la imagen subida
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Leer y preprocesar la imagen
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocessed_img = preprocess_image(img)

    # Realizar predicción
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    # Limpiar archivo temporal
    os.remove(file_path)

    # Respuesta con la predicción
    predicted_class = class_labels[predicted_class_index]
    return jsonify({
        "prediction": predicted_class,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    # Crear carpeta temporal si no existe
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True)
