"""_crear_clip.py — Recorta 6 s de _test_vehiculos.mp4 a resolución 960 px de ancho
para obtener un clip < 15 MB que pueda subirse por la ruta /analizar."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2

ORIGEN = "_test_vehiculos.mp4"
DESTINO = "_test_clip.mp4"

cap = cv2.VideoCapture(ORIGEN)
if not cap.isOpened():
    print("CLIP_ERR: no se pudo abrir", ORIGEN)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ancho0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"ORIGEN fps={fps:.2f} {ancho0}x{alto0}")

escala = 960.0 / ancho0
ancho = int(ancho0 * escala) // 2 * 2
alto = int(alto0 * escala) // 2 * 2

cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 1))  # Empieza en el segundo 1

escritor = cv2.VideoWriter(DESTINO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
objetivo = int(fps * 6)  # 6 segundos
escritos = 0
while escritos < objetivo:
    leido, fotograma = cap.read()
    if not leido:
        break
    escritor.write(cv2.resize(fotograma, (ancho, alto)))
    escritos += 1
escritor.release()
cap.release()

tamano = os.path.getsize(DESTINO)
print(f"CLIP_OK frames={escritos} {ancho}x{alto} bytes={tamano}")
print("CLIP_MENOR_15MB=" + str(tamano < 15 * 1024 * 1024))
