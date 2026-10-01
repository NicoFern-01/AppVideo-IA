"""Smoke test servido: v2.6.0 (Pantalla Completa + recorte estricto backend)."""
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"


def get(ruta):
    with urllib.request.urlopen(BASE + ruta, timeout=8) as respuesta:
        return respuesta.read().decode("utf-8", "replace")


version = {}
for _ in range(15):
    try:
        version = json.loads(get("/version"))
        break
    except Exception:
        time.sleep(1)

comprobaciones = [("GET /version = 2.6.0", version.get("version") == "2.6.0")]

js = get("/app.js")
comprobaciones += [
    ("Motor a 16 ms (setInterval tickMotor)", "setInterval(tickMotor, INTERVALO_BUCLE_MS)" in js),
    ("Nucleo estricto aplicarBucleEstricto", "function aplicarBucleEstricto" in js),
    ("Regla estricta inicio - 0.1", "ahora < inicio - TOLERANCIA_INFERIOR" in js),
    ("Tolerancia anti-bucle de 0.05 s", "TOLERANCIA_SEEK = 0.05" in js),
    ("Inyeccion play()/pause()", "inyeccionActiva = true" in js and js.count("reproductor.pause();") >= 2),
    ("Listener 'seeked'", 'addEventListener("seeked"' in js),
    ("Listener 'seeking'", 'addEventListener("seeking"' in js),
    ("SIN listener 'timeupdate'", 'addEventListener("timeupdate"' not in js),
    ("SIN alActualizarTiempo antiguo", "alActualizarTiempo" not in js),
    ("parseFloat estricto en el bucle", "parseFloat(tiempoInicio)" in js),
    ("Envio del recorte toFixed(3)", ".toFixed(3)" in js),
]

pagina = get("/")
segmento_video = pagina.split("<video")[1].split(">")[0] if "<video" in pagina else ""
comprobaciones += [
    ("Pagina con timeline-window", "timeline-window" in pagina),
    ("Video SIN controls nativos", "controls" not in segmento_video),
    ("Boton Pantalla Completa en la toolbar", 'id="btn-fullscreen"' in pagina),
    ("FS API con variantes en app.js", "alternarPantallaCompleta" in js and "requestFullscreen" in js),
    ("Motor sigue activo en fullscreen (sin clearInterval)", "clearInterval" not in js),
]

# --- Backend en disco: recorte matematico estricto de fotogramas --------------
codigo_backend = (Path(__file__).resolve().parent / "main.py").read_text(encoding="utf-8")
comprobaciones += [
    ("Backend: frame_start/frame_end por FPS con int()",
     "frame_start = max(0, min(int(inicio_s * fps)" in codigo_backend
     and "frame_end = max(frame_start, min(int(fin_s * fps)" in codigo_backend),
    ("Backend: salto directo al inicio del recorte",
     "cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))" in codigo_backend),
    ("Backend: bucle con corte duro en frame_end",
     "while contador <= frame_end" in codigo_backend
     and "> frame_end:" in codigo_backend),
    ("Backend: aviso de fragmento estricto al prompt",
     "única y exclusivamente al fragmento" in codigo_backend),
    ("Backend: sanea tiempos negativos/invertidos en /analizar",
     "tiempo_fin < tiempo_inicio" in codigo_backend),
]

fallos = [nombre for nombre, ok in comprobaciones if not ok]
for nombre, ok in comprobaciones:
    print(f"  [{'OK    ' if ok else 'FALLO '}] {nombre}")

print(f"\nRESULTADO: {len(comprobaciones) - len(fallos)}/{len(comprobaciones)} OK")
print("SMOKE_OK" if not fallos else f"SMOKE_FALLOS: {fallos}")
