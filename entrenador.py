import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Rutas de las carpetas
base_dir = "Agricultural-crops"
batch_size = 32
img_height, img_width = 150, 150

# Generadores de datos para aumentar imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",  # Cambiado a 'categorical' para múltiples clases
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Número de clases (almond, banana, cherry, rice, lemon, tomato)
num_classes = len(train_generator.class_indices)

# Construcción del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Ajustado para múltiples clases
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Cambiado a 'categorical_crossentropy'
    metrics=['accuracy']
)

# Entrenamiento del modelo
epochs = 20
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Guardar el modelo
model.save("crop_classifier_model.h5")

# Imprimir etiquetas de las clases para usarlas en predicción
print("Clases del modelo:", train_generator.class_indices)
