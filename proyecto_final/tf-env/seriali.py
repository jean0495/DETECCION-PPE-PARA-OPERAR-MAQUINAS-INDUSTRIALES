import serial
import time

# ─── CONFIGURACIÓN SERIAL ───────────────────────────────────────
PORT = "/dev/ttyUSB0"   # cambiar a /dev/ttyACM0 si es necesario
BAUDRATE = 115200

try:
    arduino = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # reset del Arduino al abrir puerto
    print("Conexión serial establecida")
except Exception as e:
    print("Error al conectar con Arduino:", e)
    exit()

# ─── FUNCIÓN DE ENVÍO ───────────────────────────────────────────
def enviar_estado(prediccion: str):
    """
    prediccion:
        with_glasses     → habilita sistema
        without_glasses  → bloquea sistema
    """

    if prediccion == "with_glasses":
        mensaje = "START\n"   # permiso de arranque

    elif prediccion == "without_glasses":
        mensaje = "STOP\n"    # bloqueo

    else:
        mensaje = "UNKNOWN\n"

    arduino.write(mensaje.encode())
    print("Enviado:", mensaje.strip())

# ─── TEST MANUAL ────────────────────────────────────────────────
if __name__ == "__main__":

    while True:
        entrada = input("Predicción (with/without/salir): ").strip().lower()

        if entrada == "with":
            enviar_estado("with_glasses")

        elif entrada == "without":
            enviar_estado("without_glasses")

        elif entrada == "salir":
            break

        else:
            print("Comando inválido")

    arduino.close()