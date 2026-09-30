"""_test_e2e.py — Prueba END-TO-END real: POST /analizar con _test_clip.mp4
(servidor FastAPI + YOLOv11 + RAG + Ollama qwen2.5vl:7b en vivo)."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIRECTORIO = Path(__file__).resolve().parent
CLIP = DIRECTORIO / "_test_clip.mp4"
RESULTADO = DIRECTORIO / "_test_e2e_result.json"

if not CLIP.is_file():
    print("E2E_ERR: falta _test_clip.mp4 (ejecuta antes _crear_clip.py)")
    sys.exit(1)

# Motor a probar: "python _test_e2e.py [local|nube]" (por defecto: local)
motor = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "local"
if motor not in ("local", "nube"):
    motor = "local"

# Ventana opcional del Trimmer: "python _test_e2e.py [local|nube] [inicio] [fin]"
# (0.0 = clip completo, igual que el frontend cuando no se tocan las barras)
tiempo_inicio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
tiempo_fin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

boundary = "----racecontrolE2E"
cabecera_video = (
    f"--{boundary}\r\n"
    'Content-Disposition: form-data; name="video"; filename="clip.mp4"\r\n'
    "Content-Type: video/mp4\r\n\r\n"
).encode("utf-8")
parte_contexto = (
    f"\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="contexto"\r\n\r\n'
    "Prueba E2E automática: valida YOLOv11 + RAG + el motor elegido en la ruta /analizar."
    f"\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="motor"\r\n\r\n'
    f"{motor}"
    f"\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="tiempo_inicio"\r\n\r\n'
    f"{tiempo_inicio}"
    f"\r\n--{boundary}\r\n"
    'Content-Disposition: form-data; name="tiempo_fin"\r\n\r\n'
    f"{tiempo_fin}"
    f"\r\n--{boundary}--\r\n"
).encode("utf-8")

cuerpo = cabecera_video + CLIP.read_bytes() + parte_contexto
peticion = urllib.request.Request(
    "http://127.0.0.1:8000/analizar",
    data=cuerpo,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

print(
    f"E2E: motor='{motor}' · ventana={tiempo_inicio}-{tiempo_fin}s · "
    f"enviando {CLIP.name} ({CLIP.stat().st_size} bytes)..."
)
inicio = time.time()
try:
    with urllib.request.urlopen(peticion, timeout=600) as respuesta:
        estado = respuesta.status
        datos = json.loads(respuesta.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    cuerpo_error = exc.read().decode("utf-8", errors="replace")[:600]
    print(f"E2E_ERR_HTTP tras {time.time() - inicio:.0f}s: {exc}")
    print(f"E2E_ERR_CUERPO: {cuerpo_error}")
    sys.exit(1)
except Exception as exc:  # noqa: BLE001
    print(f"E2E_ERR tras {time.time() - inicio:.0f}s: {exc}")
    sys.exit(1)

transcurrido = time.time() - inicio
RESULTADO.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"E2E_HTTP={estado} en {transcurrido:.0f}s (respuesta guardada en {RESULTADO.name})")
claves = ("analisis_temporal", "culpable", "justificacion_reglamentaria", "sancion_sugerida", "analisis_tecnico")
faltan = [clave for clave in claves if not datos.get(clave)]
if faltan:
    print("E2E_FALTAN_CLAVES:", faltan)
    sys.exit(1)
print("E2E_CLAVES_OK")
print("CULPABLE:", datos["culpable"][:200])
print("SANCION:", datos["sancion_sugerida"])
print("ANALISIS(220):", datos["analisis_temporal"][:220].replace("\n", " "))
print("E2E_COMPLETO_OK")
