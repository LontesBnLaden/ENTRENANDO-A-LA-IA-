# 🟩 1. IMPORTACIÓN DE LIBRERÍAS
# Estas bibliotecas permiten: trabajar con cámara (cv2), detectar manos (MediaPipe), cálculos (numpy),
# manejo del sistema (os), síntesis de voz (pyttsx3), y fecha/hora (datetime).
import cv2
import mediapipe as mp
import numpy as np
import os
import pyttsx3
from datetime import datetime

# 🕦 2. INICIALIZACIÓN DE SISTEMA
# Silencia mensajes de TensorFlow y configura la voz
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
voz = pyttsx3.init()
voz.setProperty('rate', 150)  # Velocidad de lectura de voz

# 📘 3. INICIALIZACIÓN DE MEDIAPIPE
# Configura los módulos para detección y dibujo de manos
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 🕪 4. FUNCIONES AUXILIARES

# 📌 Convierte coordenadas normalizadas (0-1) en píxeles

def coord(p, hand_landmarks, w, h):
    return int(hand_landmarks.landmark[p].x * w), int(hand_landmarks.landmark[p].y * h)

# 📌 Busca una cámara disponible (intenta hasta 3)
def obtener_camara():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Cámara {i} detectada y abierta")
            return cap
    print("❌ No se pudo abrir ninguna cámara.")
    return None

# 📌 Guarda texto con hora actual en un archivo de log
def guardar_log(texto):
    tiempo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("registro_senas.txt", "a", encoding="utf-8") as f:
        f.write(f"[{tiempo}] {texto}\n")

# 📌 Reproduce por voz el texto recibido
def decir(texto):
    voz.say(texto)
    voz.runAndWait()

# 📌 Detecta si la mano es izquierda o derecha
def etiqueta_mano(idx, mano, results):
    for clase in results.multi_handedness:
        if clase.classification[0].index == idx:
            label = clase.classification[0].label
            coords = tuple(np.multiply(
                (mano.landmark[mp_hands.HandLandmark.WRIST].x,
                 mano.landmark[mp_hands.HandLandmark.WRIST].y),
                [1920, 1080]).astype(int))
            return label, coords
    return None

# 🔴 5. BLOQUE PRINCIPAL: Captura de video y detección
cap = obtener_camara()
if not cap:
    exit()

mensaje_anterior = ""       # Guarda el último mensaje detectado
palabra_actual = ""         # Acumula letras para formar una palabra

# Configura MediaPipe para detectar hasta 1 mano
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)  # Invierte imagen tipo espejo
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        h, w, _ = image.shape

        mensaje = ""

        # 🔫 DETECCIÓN DE GESTOS Y LETRAS
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                puntos = {i: coord(i, hand_landmarks, w, h) for i in range(21)}

                # ✅ Gestos generales
                if puntos[2][1] > puntos[4][1] and all(puntos[2][1] < puntos[f][1] for f in [8, 12, 16, 20]):
                    mensaje = 'GOOD'
                elif puntos[2][1] < puntos[4][1] and all(puntos[2][1] > puntos[f][1] for f in [8, 12, 16, 20]):
                    mensaje = 'BAD'
                elif puntos[4][1] < puntos[2][1] and puntos[8][1] < puntos[6][1] and puntos[20][1] < puntos[18][1]:
                    mensaje = 'I LOVE YOU'

                # 🔤 Letras del abecedario
                elif all(puntos[i][1] > puntos[i - 2][1] for i in [8, 12, 16, 20]) and puntos[4][0] < puntos[3][0]:
                    mensaje = "Letra: A"
                elif puntos[4][1] < puntos[3][1] and puntos[8][1] < puntos[6][1] and \
                     all(puntos[i][1] > puntos[i - 2][1] for i in [12, 16, 20]):
                    mensaje = "Letra: L"
                elif puntos[4][1] < puntos[3][1] and puntos[20][1] < puntos[18][1] and \
                     all(puntos[i][1] > puntos[i - 2][1] for i in [8, 12, 16]):
                    mensaje = "Letra: Y"

                etiqueta = etiqueta_mano(idx, hand_landmarks, results)
                if etiqueta:
                    cv2.putText(image, etiqueta[0], etiqueta[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 🖑 PROCESAMIENTO DEL MENSAJE DETECTADO
        if mensaje and mensaje != mensaje_anterior:
            print(f"🖐️ Seña detectada: {mensaje}")
            guardar_log(mensaje)
            decir(mensaje)
            mensaje_anterior = mensaje

            # Añadir letra a la palabra
            if "Letra: " in mensaje:
                letra = mensaje.split(": ")[1]
                palabra_actual += letra

            # Confirmar palabra con GOOD
            elif mensaje == "GOOD" and palabra_actual:
                guardar_log(f"Palabra construida: {palabra_actual}")
                decir(f"Palabra: {palabra_actual}")
                palabra_actual = ""

            # Borrar palabra con BAD
            elif mensaje == "BAD":
                palabra_actual = ""

        # 🕦 VISUALIZACIÓN EN PANTALLA
        if mensaje:
            cv2.putText(image, mensaje, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
        if palabra_actual:
            cv2.putText(image, f"Palabra: {palabra_actual}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

        # Mostrar la ventana
        cv2.imshow('Lenguaje de Señas + Letras + Palabras', image)

        # 🔴 Cierre del programa con 'q' o ESC
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            print("🔴 Programa finalizado por el usuario.")
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
