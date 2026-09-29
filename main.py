"""
===============================================================================
 main.py — 🏁 RACE CONTROL: Backend FastAPI + IA local de visión
===============================================================================

Servidor web LOCAL que analiza incidentes ("toques") en competencias de
automovilismo y devuelve un informe de sanción en JSON:

  1. GET  "/"         -> sirve el frontend nativo (index.html + styles.css + app.js)
  2. POST "/analizar" -> recibe un clip corto (MP4/MOV, máx. 15 MB) y el
     contexto opcional de la maniobra (casilla "Contexto del Incidente"), y:
       a) extrae un máximo ESTRUCTO de 5 fotogramas distribuidos
          equitativamente (antes / durante / después del contacto);
       b) redimensiona CADA fotograma a 640x480 px con cv2.resize (obligatorio
          para no saturar la VRAM de la GPU de 6 GB);
       c) comprime a JPEG calidad 85 y convierte a Base64;
       d) envía la secuencia a Ollama con el modelo 'qwen2.5vl:7b' usando un
          System Prompt de Comisario FIA nivel F1 con Few-Shot + Chain of
          Thought (Pasos 1-3) que exige JSON estricto;
       e) devuelve {"analisis_temporal", "culpable", "justificacion_reglamentaria",
          "sancion_sugerida"} más la clave "analisis_tecnico" (compuesta) que
          consume 'app.js' sin cambios.

Requisitos:
  - python -m pip install -r requirements.txt
  - ollama serve   y   ollama pull qwen2.5vl:7b

Ejecución:
  - python main.py   (o)   python -m uvicorn main:app --reload
  - Abrir http://127.0.0.1:8000
===============================================================================
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import ollama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# ==============================================================================
# 1. CONFIGURACIÓN Y CONSTANTES GLOBALES
# ==============================================================================
DIRECTORIO_BASE = Path(__file__).resolve().parent

# --- Procesamiento de imagen (CRÍTICO para GPU de 6 GB de VRAM) ---------------
ANCHO_FOTOGRAMA = 640               # Resolución fija obligatoria
ALTO_FOTOGRAMA = 480
CALIDAD_JPEG = 85
MAX_FOTOGRAMAS = 5                  # Máximo estricto por análisis
MAX_INTENTOS_CONTEO = 100_000       # Tope de seguridad al contar frames

# --- Validación del clip subido ----------------------------------------------
TAMANO_MAXIMO_MB = 15
TAMANO_MAXIMO_BYTES = TAMANO_MAXIMO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".mp4", ".mov"}

# --- Contexto opcional (casilla "Contexto del Incidente" del frontend) --------
MAX_CONTEXTO_CHARS = 500           # Tope del texto de contexto: protege num_ctx

# --- Modelo de visión local ---------------------------------------------------
MODELO_VLM = "qwen2.5vl:7b"
URL_OLLAMA = "http://localhost:11434"

# --- System Prompt: Comisario FIA nivel F1 (CoT paso a paso + Few-Shot) -------
# La IA debe seguir un proceso de pensamiento encadenado (Pasos 1-3), reglamento
# estricto y devolver un JSON con 4 claves. Tras el prompt va un EJEMPLO FEW-SHOT
# que calibra el formato de salida sin alterar la lógica del comisario.
SYSTEM_PROMPT = """Actúas como un Comisario Deportivo de la FIA de nivel de Fórmula 1 y Competencias Internacionales. Tu tarea es analizar de forma rigurosa una secuencia temporal de 5 fotogramas clave de un incidente en pista.

Para evitar sesgos y errores, debes seguir obligatoriamente este proceso de pensamiento paso a paso en tu análisis interno:
Paso 1: Describe qué posición y trayectoria aproximada tienen los vehículos involucrados en el Fotograma 1 (frenada/aproximación).
Paso 2: Describe el movimiento en el Fotograma 2 y 3 (entrada a la curva y punto de Apex). ¿Algún auto se tiró de lejos (Divebomb)? ¿Algún auto cambió de trayectoria en zona de frenado?
Paso 3: Identifica el momento exacto del contacto (Fotograma 4 o 5) y evalúa si el auto del interior dejó suficiente espacio (mínimo el ancho de un auto) o si el auto del exterior cerró la línea de forma ilegal.

REGLAMENTO DE REFERENCIA ESTRICTO:
- Si el auto atacante va por el interior pero su eje delantero NO supera el retrovisor del auto defensor antes del vértice de la curva, el defensor tiene derecho a la línea ideal. Culpa del atacante.
- Si hay un cambio de trayectoria brusco en zona de frenada ('Moving under braking') por parte del defensor, la culpa es del defensor.
- Si el auto del interior bloquea neumáticos (frenada pasada) y arrastra al auto exterior, es una colisión evitable. Culpa del auto interior.

FORMATO DE RESPUESTA EXIGIDO (Devuelve estrictamente este JSON):
{
  "analisis_temporal": "Tu descripción detallada fotograma por fotograma según los pasos de pensamiento anteriores.",
  "culpable": "Identificación exacta del vehículo culpable (especifica color, número o si iba por dentro/fuera).",
  "justificacion_reglamentaria": "La regla exacta del reglamento que se violó para determinar dicha culpabilidad.",
  "sancion_sugerida": "Advertencia / +5 Segundos / +10 Segundos / Drive Through / Incidente de carrera"
}

--- EJEMPLO FEW-SHOT (calibra formato y profundidad; NUNCA copies sus datos) ---
Caso hipotético de entrada:
- Fotograma 1 (frenada): auto verde #33 por el interior y auto plateado #8 por fuera, alineados y a la misma distancia del vértice.
- Fotograma 2: el #33 todavía no lleva su eje delantero por delante del retrovisor del #8.
- Fotograma 3: el #33 frena tarde, bloquea neumáticos y se queda pegado al #8 (intento de divebomb).
- Fotograma 4 y 5: contacto en el vértice y ambos vehículos salen de la trazada.
Salida esperada (únicamente el JSON, sin texto adicional):
{
  "analisis_temporal": "Fotograma 1: ambos autos alineados en frenada... Fotograma 2: el #33 no supera el retrovisor... Fotograma 3: bloqueo de frenada del #33... Fotograma 4 y 5: contacto en el vértice y pérdida de trazada...",
  "culpable": "Auto verde número 33 (iba por el interior)",
  "justificacion_reglamentaria": "El atacante no superó con su eje delantero el retrovisor del defensor antes del vértice y bloqueó neumáticos: colisión evitable, culpa del auto interior.",
  "sancion_sugerida": "+5 Segundos"
}
--- FIN DEL EJEMPLO ---
Aplica exactamente este mismo proceso de pensamiento, formato y nivel de detalle al INCIDENTE REAL que se te adjunta a continuación."""

# ==============================================================================
# 2. PROCESAMIENTO DE VIDEO OPTIMIZADO (BACKEND)
# ==============================================================================

def _contar_frames_secuencial(cap: cv2.VideoCapture) -> int:
    """Cuenta frames con grab() cuando el contenedor no informa CAP_PROP_FRAME_COUNT."""
    total = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while total < MAX_INTENTOS_CONTEO and cap.grab():
        total += 1
    return total


def extraer_fotogramas_base64(ruta_video: str, max_frames: int = MAX_FOTOGRAMAS) -> List[str]:
    """
    Extrae como máximo `max_frames` fotogramas distribuidos EQUITATIVAMENTE a lo
    largo de la duración total del clip (captura el antes, durante y después del
    choque) y los devuelve como lista de cadenas Base64.

    Optimización para GPU de 6 GB de VRAM:
      * cv2.resize OBLIGATORIO a 640x480 px por fotograma.
      * Compresión JPEG calidad 85 con cv2.imencode.
      * Conversión a Base64 recién cuando el JPEG ya está reducido.

    Lanza ValueError si el video no se puede abrir o no contiene fotogramas.
    """
    if not os.path.isfile(ruta_video):
        raise ValueError("No se encontró el archivo de video temporal.")

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video. Verifica que sea un clip MP4/MOV válido.")

    try:
        # --- 1) Duración total del clip ----------------------------------------
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = _contar_frames_secuencial(cap)
        if total_frames <= 0:
            raise ValueError("El video no contiene fotogramas legibles.")

        # --- 2) Índices equidistantes (del primer al último fotograma) ---------
        cantidad = min(max_frames, total_frames)
        indices = np.unique(np.linspace(0, total_frames - 1, cantidad).round().astype(int))

        # --- 3) Extracción -> resize 640x480 -> JPEG 85 -> Base64 --------------
        fotogramas: List[str] = []
        for indice in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(indice))
            leido, imagen = cap.read()
            if not leido or imagen is None:
                continue

            imagen = cv2.resize(
                imagen,
                (ANCHO_FOTOGRAMA, ALTO_FOTOGRAMA),
                interpolation=cv2.INTER_AREA,
            )

            codificado, buffer = cv2.imencode(
                ".jpg",
                imagen,
                [int(cv2.IMWRITE_JPEG_QUALITY), CALIDAD_JPEG],
            )
            if not codificado:
                continue

            fotogramas.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))

        if not fotogramas:
            raise ValueError("No se pudo extraer ningún fotograma del video.")
        return fotogramas
    finally:
        cap.release()


# ==============================================================================
# 3. INTEGRACIÓN CON OLLAMA (MODELO DE VISIÓN LOCAL)
# ==============================================================================

def _normalizar_sancion(valor: str) -> str:
    """Reconoce variantes ('5s', 'drive-through'...) y devuelve la forma canónica."""
    limpio = valor.strip().lower().replace(" ", "")
    equivalencias = {
        "advertencia": "Advertencia",
        "warning": "Advertencia",
        "+5s": "+5s",
        "5s": "+5s",
        "+5": "+5s",
        "+5segundos": "+5s",     # Variante del nuevo prompt ("+5 Segundos")
        "5segundos": "+5s",
        "+10s": "+10s",
        "10s": "+10s",
        "+10": "+10s",
        "+10segundos": "+10s",   # Variante del nuevo prompt ("+10 Segundos")
        "10segundos": "+10s",
        "drivethrough": "Drive Through",
        "drive-through": "Drive Through",
        "drive_through": "Drive Through",
        "paseporboxes": "Drive Through",
        "incidentedecarrera": "Incidente de carrera",
        "incidente": "Incidente de carrera",
    }
    normalizada = equivalencias.get(limpio)
    if normalizada:
        return normalizada
    # Degradación controlada: el frontend espera SIEMPRE una sanción del catálogo
    # para aplicar el color; ante "No determinado" (o vacío) se asume la más leve.
    if limpio in {"", "nodeterminado", "nodeterminada"}:
        return "Advertencia"
    return valor.strip()


def _construir_mensaje_usuario(contexto: str | None) -> str:
    """
    Construye el User Message para Ollama inyectando dinámicamente el texto
    opcional que el usuario escribe en la casilla "Contexto del Incidente".

    Estructura exigida:
      "Aquí tienes los 5 fotogramas clave del incidente. CONTEXTO ADICIONAL
       APORTADO POR LOS COMISARIOS EN PISTA: [contexto]. Por favor, procesa el
       análisis y devuelve estrictamente la estructura JSON requerida."
    """
    texto = (contexto or "").strip()
    if len(texto) > MAX_CONTEXTO_CHARS:
        texto = texto[:MAX_CONTEXTO_CHARS]
    texto = texto.rstrip(" .") or "Ninguno aportado"
    return (
        "Aquí tienes los 5 fotogramas clave del incidente. "
        "CONTEXTO ADICIONAL APORTADO POR LOS COMISARIOS EN PISTA: "
        f"{texto}. "
        "Por favor, procesa el análisis y devuelve estrictamente la "
        "estructura JSON requerida."
    )


def generar_informe_json(
    fotogramas_base64: List[str],
    contexto: str | None = None,
) -> Dict[str, str]:
    """
    Envía la secuencia de fotogramas Base64 a Ollama (`ollama.chat`) aplicando
    Few-Shot Prompting + Chain of Thought (ver SYSTEM_PROMPT) y devuelve el
    informe estructurado con las claves:
      * analisis_temporal           -> CoT: descripción fotograma por fotograma
      * culpable                    -> identificación del vehículo culpable
      * justificacion_reglamentaria -> regla del reglamento violada
      * sancion_sugerida            -> SIEMPRE una opción del catálogo
      * analisis_tecnico            -> clave COMPATIBLE con 'app.js' (compuesta
                                       por las dos piezas anteriores)

    Parámetros
    ----------
    fotogramas_base64 : list[str]
        Fotogramas JPEG 640x480 (calidad 85) en Base64, en orden cronológico.
    contexto : str | None
        Texto opcional del usuario ("Contexto del Incidente") que se inyecta
        dinámicamente en el User Message para guiar el análisis visual.

    Lanza HTTPException ante fallos de conexión, modelo ausente o JSON inválido.
    """
    if not fotogramas_base64:
        raise HTTPException(status_code=400, detail="No hay fotogramas para analizar.")

    # --- User Message: estructura exacta con contexto inyectado ---------------
    mensaje_usuario = _construir_mensaje_usuario(contexto)

    try:
        respuesta = ollama.chat(
            model=MODELO_VLM,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje_usuario, "images": fotogramas_base64},
            ],
            format="json",           # Ollama fuerza salida JSON válida
            keep_alive="1h",         # Mantiene el modelo residente: evita la
                                     # recarga (~2-4 min) entre análisis seguidos
            options={
                "temperature": 0.2,  # Criterio "reglamentario" y repetible
                "num_predict": 900,  # Holgura para el análisis temporal detallado
                # Presupuesto: ~5.300 tokens de imágenes + ~1.400 del prompt
                # CoT/Few-Shot ≈ 6.700; con num_ctx 8192 queda margen para la
                # salida. (El num_ctx por defecto de Ollama, 4096, daría error 400.)
                "num_ctx": 8192,
            },
        )
    except ollama.ResponseError as exc:
        estado = getattr(exc, "status_code", None)
        if estado == 404:
            raise HTTPException(
                status_code=502,
                detail=f"El modelo '{MODELO_VLM}' no está instalado. Ejecuta: ollama pull {MODELO_VLM}",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Ollama devolvió un error HTTP {estado}: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # Ollama apagado / sin conexión
        raise HTTPException(
            status_code=503,
            detail=f"No responde Ollama en {URL_OLLAMA}. Ejecuta 'ollama serve'. Detalle: {exc}",
        ) from exc

    contenido = (respuesta.get("message", {}) or {}).get("content", "").strip()
    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="La IA devolvió una respuesta que no es JSON válido.",
        ) from exc
    if not isinstance(datos, dict):
        raise HTTPException(status_code=502, detail="La respuesta de la IA no es un objeto JSON.")

    # --- Nuevas claves del prompt CoT + clave histórica para app.js ------------
    analisis_temporal = str(datos.get("analisis_temporal") or "").strip()
    justificacion = str(datos.get("justificacion_reglamentaria") or "").strip()
    culpable = str(datos.get("culpable") or "").strip() or "No determinado"
    sancion = _normalizar_sancion(str(datos.get("sancion_sugerida") or ""))

    # 'app.js' lee "analisis_tecnico" (tarjeta "2) ANÁLISIS TÉCNICO"): se compone
    # con las dos piezas nuevas para mantener el contrato JSON intacto.
    if analisis_temporal or justificacion:
        partes = []
        if analisis_temporal:
            partes.append(f"ANÁLISIS TEMPORAL:\n{analisis_temporal}")
        if justificacion:
            partes.append(f"JUSTIFICACIÓN REGLAMENTARIA:\n{justificacion}")
        analisis_tecnico = "\n\n".join(partes)
    else:
        # Compatibilidad si el modelo devolviera la clave antigua "analisis_tecnico"
        analisis_tecnico = str(datos.get("analisis_tecnico") or "").strip() or "No disponible."

    return {
        "analisis_temporal": analisis_temporal or "No disponible.",
        "culpable": culpable,
        "justificacion_reglamentaria": justificacion or "No disponible.",
        "sancion_sugerida": sancion,
        "analisis_tecnico": analisis_tecnico,  # clave que consume app.js
    }


# ==============================================================================
# 4. RUTAS DEL SERVIDOR
# ==============================================================================

def guardar_clip_temporal(datos: bytes, extension: str) -> str:
    """Persiste el clip subido en un archivo temporal con la extensión validada."""
    descriptor, ruta = tempfile.mkstemp(prefix="racecontrol_", suffix=extension)
    with os.fdopen(descriptor, "wb") as archivo:
        archivo.write(datos)
    return ruta


app = FastAPI(
    title="Race Control API",
    description="Backend local para análisis de incidentes en pista con IA de visión (Ollama).",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def servir_index() -> FileResponse:
    """Sirve el frontend nativo (index.html)."""
    return FileResponse(DIRECTORIO_BASE / "index.html", media_type="text/html; charset=utf-8")


@app.get("/styles.css", include_in_schema=False)
def servir_estilos() -> FileResponse:
    """Sirve la hoja de estilos (styles.css)."""
    return FileResponse(DIRECTORIO_BASE / "styles.css", media_type="text/css; charset=utf-8")


@app.get("/app.js", include_in_schema=False)
def servir_script() -> FileResponse:
    """Sirve el script de interactividad (app.js)."""
    return FileResponse(DIRECTORIO_BASE / "app.js", media_type="application/javascript; charset=utf-8")


@app.post("/analizar")
async def analizar_incidente(
    video: UploadFile = File(..., description="Clip corto del incidente (MP4/MOV, máx. 15 MB)"),
    contexto: str | None = Form(
        None,
        description="Contexto opcional de la maniobra (atacante/defensor, números/colores)",
    ),
) -> JSONResponse:
    """
    Pipeline completo: valida el clip -> extrae 5 fotogramas (640x480 / JPEG 85)
    -> consulta al modelo de visión local -> devuelve el JSON del informe.
    """
    # (1) Formato del archivo
    extension = Path(video.filename or "").suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(status_code=400, detail="Formato no admitido. Envía un archivo .mp4 o .mov.")

    # (2) Peso máximo 15 MB (lectura acotada: no se carga nada más en memoria)
    datos = await video.read(TAMANO_MAXIMO_BYTES + 1)
    if not datos:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(datos) > TAMANO_MAXIMO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"El clip supera los {TAMANO_MAXIMO_MB} MB. Recorta un clip de "
                "5 a 10 segundos con el momento exacto del toque."
            ),
        )

    # (3) Extracción de fotogramas optimizados
    ruta_temporal = guardar_clip_temporal(datos, extension)
    try:
        fotogramas = extraer_fotogramas_base64(ruta_temporal, MAX_FOTOGRAMAS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(ruta_temporal)

    # (4) Análisis con IA local y (5) respuesta JSON directa al frontend
    return JSONResponse(content=generar_informe_json(fotogramas, contexto))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
