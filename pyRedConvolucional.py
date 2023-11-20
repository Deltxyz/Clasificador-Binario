from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Configuración para la carga de archivos
app.config['UPLOAD_FOLDER'] = 'img'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Cargar el modelo preentrenado
modelo = load_model('modelo_perro_gato.h5')  # Asegúrate de tener un modelo entrenado disponible

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clasificar_imagen(ruta_imagen):
    # Cargar y preprocesar la imagen
    img = load_img(ruta_imagen, target_size=(128, 128))  # Ajusta el tamaño de la imagen según tu modelo
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalizar los valores de píxeles al rango [0, 1]
    img_array /= 255.0

    # Realizar la predicción
    prediccion = modelo.predict(img_array)

    # Devolver el resultado
    return prediccion[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None

    if request.method == 'POST':
        # Verificar si se ha enviado un archivo
        if 'imagen' not in request.files:
            return render_template('index.html', resultado=resultado)

        file = request.files['imagen']

        # Verificar si se ha seleccionado un archivo y es válido
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', resultado=resultado)

        # Guardar el archivo en la carpeta 'img'
        filename = secure_filename(file.filename)
        ruta_imagen = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(ruta_imagen)

        # Realizar la clasificación
        resultado = clasificar_imagen(ruta_imagen)

    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
