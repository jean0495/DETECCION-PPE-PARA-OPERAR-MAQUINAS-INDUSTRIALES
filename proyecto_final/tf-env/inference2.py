import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# ─── Configuración ───────────────────────────────────────────────
MODEL_PATH   = "/home/jean/Desktop/proyecto_final/tf-env/my_model_quantized.tflite"
CASCADE_PATH = "/home/jean/Desktop/proyecto_final/haarcascade_frontalface_default.xml"
CLASS_NAMES  = ["with_glasses", "without_glasses"]
SERIAL_PORT  = "/dev/ttyUSB0"
BAUDRATE     = 115200

# ─── Serial ──────────────────────────────────────────────────────
arduino = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
time.sleep(2)
print("Serial conectado")

# ─── Modelo ──────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ─── Cascade ─────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ─── Cámara ──────────────────────────────────────────────────────
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: no se pudo abrir la cámara")
    exit()

print("Presiona ESPACIO para clasificar | q para salir")

ultimo_estado = None

# ─── Funciones ───────────────────────────────────────────────────
def enviar(msg):
    arduino.write((msg + "\n").encode())
    print("Enviado:", msg)

def clasificar(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    img = np.array(img, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    pred   = np.argmax(output, axis=1)[0]
    return CLASS_NAMES[pred]

def dibujar_rostro(frame, prediccion):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    color = (0, 255, 0) if prediccion == "with_glasses" else (0, 0, 255)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    return frame

# ─── Loop principal ──────────────────────────────────────────────
while True:

    ret, frame = cam.read()
    if not ret:
        print("Error al leer frame")
        break

    cv2.imshow("Camara", frame)

    # ─── Teclas desde la ventana OpenCV ──────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Saliendo...")
        break

    elif key == 32:  # ESPACIO → clasificar

        prediccion = clasificar(frame)
        print("Predicción:", prediccion)

        resultado = dibujar_rostro(frame.copy(), prediccion)

        label = "Con gafas" if prediccion == "with_glasses" else "Sin gafas"
        cv2.putText(resultado, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if prediccion == "with_glasses" else (0, 0, 255),
                    2)

        cv2.imshow("Resultado", resultado)

        # Enviar por serial solo si cambia el estado
        if prediccion != ultimo_estado:
            if prediccion == "with_glasses":
                enviar("START")
            else:
                enviar("STOP")
            ultimo_estado = prediccion

# ─── Cierre ──────────────────────────────────────────────────────
cam.release()
arduino.close()
cv2.destroyAllWindows()