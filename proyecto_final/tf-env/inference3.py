import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import sys
import select

arduino = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
time.sleep(2)
print("Serial conectado")

MODEL_PATH = "/home/jean/Desktop/proyecto_final/tf-env/my_model_quantized.tflite"
CLASS_NAMES = ["with_glasses", "without_glasses"]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error cámara")
    exit()

print("Presiona ENTER en terminal para clasificar (q para salir)")

ultimo_estado = None

def enviar(msg):
    arduino.write((msg + "\n").encode())
    print("Enviado:", msg)

def key_pressed():
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.readline().strip()
    return None

while True:

    ret, frame = cam.read()
    if not ret:
        break

    # detectar tecla en terminal
    key = key_pressed()

    if key == "q":
        break

    if key == "":  # ENTER
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        img = np.array(img, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output, axis=1)[0]

        prediccion = CLASS_NAMES[pred]

        print("Predicción:", prediccion)

        if prediccion != ultimo_estado:

            if prediccion == "with_glasses":
                enviar("START")
            else:
                enviar("STOP")

            ultimo_estado = prediccion

cam.release()
arduino.close()