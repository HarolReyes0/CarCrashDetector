from flask import Flask, request, render_template, send_from_directory
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)  # Crear una aplicación Flask

@app.route("/", methods=['GET', 'POST'])  # Definir la ruta principal de la aplicación
def home():
    # Manejar las solicitudes GET y POST en la ruta principal
    if request.method == 'POST':  # Verificar si la solicitud es de tipo POST
        # Obtener la imagen enviada desde el formulario
        image = request.files['imagefile']
        image_path = "Images/" + image.filename  # Ruta donde se guarda la imagen
        image.save(image_path)  # Guardar la imagen en el servidor

        # Leer la imagen utilizando OpenCV
        img = cv2.imread(image_path)
        # Redimensionar la imagen a 256x256 píxeles
        resize = cv2.resize(img, (256, 256))
        # Cargar el modelo de clasificación de imágenes
        model = load_model('model\\imageclassifier.h5')
        # Predecir la clase de la imagen utilizando el modelo
        y_hat = model.predict(np.expand_dims(resize / 255, 0))

        # Determinar la clasificación en funció n de la predicción
        if y_hat > 0.5:
            classification = y_hat  # Si la predicción es mayor que 0.5, clasificar como "siniestrado"
        else:
            classification = y_hat  # De lo contrario, clasificar como "sin siniestrar"

        # Renderizar la plantilla HTML con la predicción y la ruta de la imagen
        return render_template('index.html', prediction=classification, image_path=image_path)

    # Si la solicitud es de tipo GET, simplemente renderizar la plantilla HTML
    return render_template('index.html')

@app.route('/display/<filename>')  # Definir la ruta para mostrar la imagen
def display_image(filename):
    # Enviar la imagen especificada desde el directorio "Images" al cliente
    return send_from_directory("Images", filename)

if __name__ == '__main__':
    app.run(debug=True)  # Iniciar la aplicación Flask en modo de depuración
