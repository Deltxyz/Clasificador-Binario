import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración del modelo
def build_model():
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(128, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(1, activation='sigmoid'))
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

# Configuración del generador de imágenes para aumentar datos
def create_data_generator():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    return train_datagen

# Especifica la ruta del directorio que contiene las imágenes de perros y gatos
ruta_directorio_entrenamiento = 'dataset'

# Configura el generador de imágenes de entrenamiento
def create_data_flow(train_datagen):
    train_generator = train_datagen.flow_from_directory(
        ruta_directorio_entrenamiento,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    return train_generator

# Entrenamiento del modelo
def train_model(model, train_generator, epochs, steps_per_epoch):
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Guardar el modelo entrenado con información adicional
def save_model(model, model_name):
    model.save(model_name + '.h5')
    model_json = model.to_json()
    with open(model_name + "_architecture.json", "w") as json_file:
        json_file.write(model_json)

# Configuración y entrenamiento del modelo
epochs = 100
model = build_model()
train_datagen = create_data_generator()
train_generator = create_data_flow(train_datagen) 
steps_per_epoch = len(train_generator)
train_model(model, train_generator, epochs, steps_per_epoch)

# Guardar el modelo entrenado con información adicional
save_model(model, 'modelo_perro_gato')
