import numpy as np
import cv2
import tensorflow as tf

# ─── Configuración ───────────────────────────────────────────────
MODEL_PATH   = "/home/jean/Desktop/proyecto_final/tf-env/my_model_quantized.tflite"
CASCADE_PATH = "/home/jean/Desktop/proyecto_final/haarcascade_frontalface_default.xml"
CLASS_NAMES  = ["with_glasses", "without_glasses"]

# ─── Cargar modelo y cascade ─────────────────────────────────────
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
interpreter  = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ─── Cámara ──────────────────────────────────────────────────────
cam = cv2.VideoCapture(0)
cv2.namedWindow("camara")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error al leer frame")
        break

    cv2.imshow("camara", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:       # ESC → salir
        print("Cerrando...")
        break

    elif k % 256 == 32:     # ESPACIO → capturar
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} guardada!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

# ─── Inferencia sobre la última foto ─────────────────────────────
img = cv2.imread("frame_0.png")
img2 = cv2.resize(img, (224, 224))
img2 = np.reshape(img2, [1, 224, 224, 3])
img2 = img2 / 255.0
img2 = np.array(img2, dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], img2)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])
prediction_classes = np.argmax(predictions, axis=1)
prediccion = CLASS_NAMES[prediction_classes[0]]
print("Predicción:", prediccion)

# ─── Detectar rostro y dibujar rectángulo ────────────────────────
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    if prediccion == "with_glasses":
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("Resultado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()