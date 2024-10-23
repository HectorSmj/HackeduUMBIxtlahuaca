import cv2
import numpy as np
import tensorflow as tf
import os

# Ruta al archivo del modelo
model_path = os.path.expanduser("C:/Users/alan8/Desktop/aguacates/modelo_aguacate.h5")

# Verificar si el modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError("No se encontró el modelo en la ruta especificada.")

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Clases del modelo
class_names = ['inmaduro', 'maduro', 'pasado']

# Iniciar la captura de video
cap = cv2.VideoCapture(0)  # Cambiar si es necesario

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el frame.")
        break

    img_resized = cv2.resize(frame, (150, 150))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_names[predicted_class]
    except Exception as e:
        print(f"Error en la predicción: {e}")
        break

    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Aguacate Clasificador', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
