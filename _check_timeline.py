"""Comprobación rápida del rediseño Timeline (temporal)."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = Path(__file__).resolve().parent
js = (base / "app.js").read_text(encoding="utf-8")
html = (base / "index.html").read_text(encoding="utf-8")
css = (base / "styles.css").read_text(encoding="utf-8")

# 1) Todos los getElementById de app.js existen en index.html
ids = sorted(set(re.findall(r"getElementById\(\"([\w-]+)\"\)", js)))
missing = [i for i in ids if f'id="{i}"' not in html]
print(f"IDs referenciados: {len(ids)} | Faltan: {missing or 'NINGUNO'}")

# 2) HTML bien formado
p = HTMLParser()
p.feed(html)
p.close()
print("HTML parse: OK")

# 3) Sin restos del trimmer antiguo
restos = [
    t for t in ("trimmer", "dual-range", "rango-inicio", "rango-fin",
                "alMoverRango", "suprimirBucle", "TOLERANCIA_BUCLE",
                "formatearTiempoLargo", "actualizarTrimmer",
                "ultimoSaltoForzado =", "TOLERANCIA_REJILLA", "alActualizarTiempo")
    if t in js or t in html or t in css
]
print("Restos trimmer:", restos or "NINGUNO")

# 4) Contenido nuevo presente
requerido_html = [
    "timeline-window", "timeline-ruler", "timeline-tracks", "selection-overlay",
    "playhead-line", "timeline-play", "btn-quitar", "Video Track 1",
    "btn-fullscreen",
]
requerido_js = [
    "tiempoInicio", "tiempoFin", "construirRegla", "aplicarBucleEstricto",
    "inyectarFotograma", "INTERVALO_BUCLE_MS", "alIniciarArrastre", "alClicRegla", "toFixed(3)", "bucleAguja",
    "alternarPantallaCompleta", "requestFullscreen",
]
faltan_h = [x for x in requerido_html if x not in html]
faltan_j = [x for x in requerido_js if x not in js]
print("HTML nuevo falta:", faltan_h or "NINGUNO")
print("JS nuevo falta:", faltan_j or "NINGUNO")

# 5) <video> SIN controles nativos
video_tag = re.search(r"<video[^>]*>", html).group(0)
print("video tag:", video_tag)
print("¿controls presente?:", "controls" in video_tag)

# 6) Llaves CSS balanceadas
print("CSS llaves:", css.count("{"), "/", css.count("}"))
