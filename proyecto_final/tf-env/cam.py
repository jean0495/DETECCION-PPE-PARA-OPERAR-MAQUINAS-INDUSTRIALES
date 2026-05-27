import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

print("✅ Cámara abierta. Presiona ESC para cerrar.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Error al leer frame")
        break

    cv2.imshow("Camara USB", frame)

    if cv2.waitKey(1) % 256 == 27:  # ESC para salir
        break

cam.release()
cv2.destroyAllWindows()