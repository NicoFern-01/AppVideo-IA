"""_test_gemini_key.py — Verifica la integración VIVA con la API de Gemini:
envía una petición mínima (sin imágenes) y muestra la respuesta o el error."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main  # noqa: E402  (carga constantes y funciones del backend)

print(f"Modelo: {main.MODELO_GEMINI}")
print(
    f"Clave (parcial): {main.CLAVE_GEMINI[:8]}...{main.CLAVE_GEMINI[-6:]} "
    f"({len(main.CLAVE_GEMINI)} caracteres)"
)

# --- Chequeo offline del guard: sin clave, el motor Nube debe fallar claro ---
clave_original = main.CLAVE_GEMINI
main.CLAVE_GEMINI = ""
try:
    try:
        main._consultar_gemini("x", [])
        print("GUARD_FALLO: no lanzó error con clave vacía")
    except Exception as exc:  # HTTPException esperada
        print("GUARD_OK:", str(getattr(exc, "detail", exc))[:90])
finally:
    main.CLAVE_GEMINI = clave_original

try:
    texto = main._consultar_gemini('Responde únicamente con el JSON {"ok": true}.', [])
except Exception as exc:  # HTTPException u otros fallos de red
    detalle = getattr(exc, "detail", None) or str(exc)
    print("GEMINI_ERR:", detalle)
    sys.exit(1)

print("GEMINI_RESPUESTA:", texto[:200])
print("GEMINI_OK")
