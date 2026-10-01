"""
===============================================================================
 main.py — 🏁 RACE CONTROL: Backend FastAPI + IA local de visión
===============================================================================

Servidor web LOCAL que analiza incidentes ("toques") en competencias de
automovilismo y devuelve un informe de sanción en JSON:

  1. GET  "/"         -> sirve el frontend nativo (index.html + styles.css + app.js)
  2. POST "/analizar" -> recibe un clip (MP4/MOV), el contexto opcional de la
     maniobra y el MOTOR elegido ("local" = Ollama, "nube" = Gemini), y:
       a) extrae fotogramas equidistantes según el motor: máx. 5 en local
          (antes / durante / después del contacto) y hasta 20 en nube, para
          clips de 20 a 45 s sin perder la resolución temporal del toque, siempre
          dentro de la ventana seleccionada en el Trimmer (inicio/fin);
       b) redimensiona CADA fotograma a 640x480 px con cv2.resize (obligatorio
          para no saturar la VRAM de la GPU de 6 GB);
       b2) pasa el fotograma 640x480 por YOLOv11-n ('yolo11n.pt') y construye la
          LECTURA DE MOVIMIENTO: aproximación/alejamiento entre los 2 vehículos
          principales, traducida a lenguaje deportivo (nada técnico copiable);
       c) comprime a JPEG calidad 85 y convierte a Base64;
       d) envía la secuencia al motor elegido — Ollama local ('qwen2.5vl:7b')
          o Gemini en la nube ('gemini-2.5-flash') — usando un System Prompt de
          Comisario FIA nivel F1 con Few-Shot + Chain of Thought (Pasos 1-3) y
          un User Message con TRES fuentes de verdad: reglamento.txt (RAG) +
          lectura de movimiento YOLO + contexto del usuario;
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
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import ollama
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# --- Visión por computadora: YOLOv11 (detección geométrica de vehículos) ------
# La importación está protegida para que la app siga arrancando (modo solo-VLM)
# si el entorno no tuviera torch/ultralytics instalados.
try:
    from ultralytics import YOLO
except Exception as _exc_import_yolo:  # pragma: no cover - ruta defensiva
    YOLO = None  # type: ignore[assignment]
    _ERROR_IMPORT_YOLO: Exception | None = _exc_import_yolo
else:
    _ERROR_IMPORT_YOLO = None

# ==============================================================================
# 1. CONFIGURACIÓN Y CONSTANTES GLOBALES
# ==============================================================================
DIRECTORIO_BASE = Path(__file__).resolve().parent


def _cargar_env_local(ruta_env: Path) -> None:
    """
    Carga pares CLAVE=VALOR desde '.env' (si existe) hacia os.environ, sin
    dependencias externas. Las variables ya definidas en el sistema tienen
    prioridad. Permite compartir el repositorio SIN subir secretos: el '.env'
    está ignorado por git (ver '.env.example').
    """
    try:
        contenido = ruta_env.read_text(encoding="utf-8")
    except OSError:
        return
    for linea in contenido.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        clave, valor = clave.strip(), valor.strip().strip('"').strip("'")
        if clave and valor and clave not in os.environ:
            os.environ[clave] = valor


_cargar_env_local(DIRECTORIO_BASE / ".env")

# --- Procesamiento de imagen --------------------------------------------------
ANCHO_FOTOGRAMA = 640               # Resolución fija obligatoria
ALTO_FOTOGRAMA = 480
CALIDAD_JPEG = 85
MAX_INTENTOS_CONTEO = 100_000       # Tope de seguridad al contar frames

# --- Motores de análisis y DENSIDAD DE FOTOGRAMAS según el motor --------------
# El motor elegido en la interfaz decide cuántos fotogramas se extraen y cuánto
# puede pesar el clip: "local" (Ollama, cuida los 6 GB de VRAM) o "nube"
# (Gemini, exprime la resolución temporal de clips de 20 a 45 segundos).
MOTOR_LOCAL = "local"
MOTOR_NUBE = "nube"
MOTORES_VALIDOS = {MOTOR_LOCAL, MOTOR_NUBE}
MAX_FOTOGRAMAS_LOCAL = 5            # Máximo estricto para el motor local
MAX_FOTOGRAMAS_NUBE = 20            # Máximo para la nube (15-20 recomendado)
MAX_FOTOGRAMAS = MAX_FOTOGRAMAS_LOCAL  # Alias histórico (retrocompatibilidad)

# --- Validación del clip subido (límite según motor) --------------------------
TAMANO_MAXIMO_MB_LOCAL = 15         # Local: clips cortos de 5-10 s
TAMANO_MAXIMO_MB_NUBE = 100         # Nube: videos de 20-45 s con libertad
EXTENSIONES_PERMITIDAS = {".mp4", ".mov"}

# --- Motor en línea: Gemini (solo se usa si el usuario elige "Nube") ----------
MODELO_GEMINI = "gemini-2.5-flash"  # Visión + salida JSON: rápido y económico
URL_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
# La clave NUNCA va en el código: se configura con la variable de entorno
# GEMINI_API_KEY (recomendado: archivo '.env' local, ignorado por git; ver
# '.env.example') o directamente en el sistema. Si falta, el motor Nube
# responde con un error claro en vez de fallar de forma silenciosa.
CLAVE_GEMINI = os.environ.get("GEMINI_API_KEY", "")

# --- Contexto opcional (casilla "Contexto del Incidente" del frontend) --------
MAX_CONTEXTO_CHARS = 500           # Tope del texto de contexto: protege num_ctx

# --- Versión del desarrollo (ÚNICA FUENTE DE VERDAD del versionado) -----------
# CÓMO VERSIONAR: al cerrar una mejora, sube VERSION_APP siguiendo SemVer
# (MAYOR cambia de arquitectura · MENOR nueva capacidad · PARCHE correcciones)
# y actualiza FECHA_ACTUALIZACION. El pie de la web lo muestra al instante
# (GET /version + app.js), así sabrás siempre qué build está corriendo.
VERSION_APP = "2.6.0"               # 2.6.0 = Pantalla Completa en la timeline + recorte
                                    #         estricto matemático de fotogramas (int(fps*t))
                                    # 2.5.1 = Motor de reproducción unificado: bucle estricto
                                    #         a 16 ms (sin timeupdate) + inyección de fotograma
                                    # 2.5.0 = Timeline window estilo NLE (regla milimétrica,
                                    #         bloque verde al ms, aguja roja, bucle estricto)
                                    # 2.4.0 = Gemini JSON STRICT + bucle acotado +
                                    #         rango dual premium (aguamarina + ms + Play/Pausa)
                                    # 2.3.0 = Trimmer (recorte por tiempo) + Gemini con
                                    #         responseSchema nativo + informe seguro por defecto
                                    # 2.2.0 = doble motor Local/Nube (Gemini) + densidad
                                    #         dinámica de fotogramas (5 local / 20 nube)
                                    # 2.1.0 = informe deportivo limpio (sin jerga técnica)
                                    # 2.0.0 = arquitectura híbrida VLM + YOLOv11 + RAG
FECHA_ACTUALIZACION = "2026-10-01"

# --- Modelo de visión local ---------------------------------------------------
MODELO_VLM = "qwen2.5vl:7b"
URL_OLLAMA = "http://localhost:11434"

# --- Visión por computadora (YOLOv11-n, nativa en Python vía ultralytics) -----
MODELO_YOLO = "yolo11n.pt"                      # Pesos ligeros (~5 MB, descarga única)
CONFIANZA_YOLO = 0.25                           # Umbral mínimo de confianza
CLASES_VEHICULO = {2: "auto", 7: "truck/kart"}  # Clases COCO nativas de vehículos
UMBRAL_TENDENCIA_PX = 3                         # ±px para Convergente/Divergente

# --- System Prompt: Comisario FIA nivel F1 (CoT paso a paso + Few-Shot) -------
# La IA debe seguir un proceso de pensamiento encadenado (Pasos 1-3), reglamento
# estricto y devolver un JSON con 4 claves. Tras el prompt va un EJEMPLO FEW-SHOT
# que calibra el formato de salida sin alterar la lógica del comisario.
SYSTEM_PROMPT = """Actúas como un Comisario Deportivo de la FIA de nivel de Fórmula 1 y Competencias Internacionales. Tu tarea es analizar de forma rigurosa una secuencia temporal de fotogramas clave de un incidente en pista (de 5 a 20, siempre en orden cronológico).

El mensaje del usuario te entrega TRES FUENTES DE VERDAD que debes cruzar obligatoriamente: (1) el REGLAMENTO OFICIAL (RAG), (2) la LECTURA DE MOVIMIENTO DE LOS VEHÍCULOS (resumen interno fotograma a fotograma: aproximación, alejamiento y contactos entre los dos autos) y (3) el CONTEXTO APORTADO POR EL USUARIO. Las imágenes confirman o matizan la lectura de movimiento; el reglamento dicta la culpa.

Para evitar sesgos y errores, debes seguir obligatoriamente este proceso de pensamiento paso a paso en tu análisis interno:
Paso 1: Describe qué posición y trayectoria aproximada tienen los vehículos al inicio (frenada/aproximación) y contrástalo con la LECTURA DE MOVIMIENTO (¿confirma que se acercaban entre sí?).
Paso 2: Describe el movimiento en el tramo medio (entrada a la curva y punto de Apex) apoyándote en las tendencias de la LECTURA DE MOVIMIENTO: ¿Algún auto se tiró de lejos (Divebomb)? ¿Algún auto cambió de trayectoria en zona de frenado?
Paso 3: Identifica el momento exacto del contacto (aviso de solapamiento en la lectura de movimiento y evidencia visual) y evalúa si el auto del interior dejó suficiente espacio (mínimo el ancho de un auto) o si el auto del exterior cerró la línea de forma ilegal.

REGLAMENTO DE REFERENCIA ESTRICTO (criterios de aplicación; el texto literal está en la FUENTE 1 adjunta):
- Si el auto atacante va por el interior pero su eje delantero NO supera el retrovisor del auto defensor antes del vértice de la curva, el defensor tiene derecho a la línea ideal. Culpa del atacante.
- Si hay un cambio de trayectoria brusco en zona de frenada ('Moving under braking') por parte del defensor, la culpa es del defensor.
- Si el auto del interior bloquea neumáticos (frenada pasada) y arrastra al auto exterior, es una colisión evitable. Culpa del auto interior.
- Catálogo de sanciones: Incidente de carrera (sin infracción clara) / Advertencia (toque leve) / +5s (colisión evitable, no respetar el ancho mínimo) / +10s o Drive Through (temerario, 'Moving under braking', empujón deliberado).
Tu 'justificacion_reglamentaria' debe CITAR el artículo exacto del reglamento adjunto (por ejemplo: ARTÍCULO 1, ARTÍCULO 2, ARTÍCULO 3...) y apoyarse en la evidencia visual y en la lectura de movimiento, siempre en lenguaje deportivo.

REGLA DE ESTILO DEL INFORME (OBLIGATORIA E INMUTABLE):
REGLA DE FORMATO OBLIGATORIA: Queda estrictamente prohibido incluir datos internos del código o de YOLO en el informe final visible. No uses palabras como "conf", números decimales de confianza (ej. 0.31), píxeles, ni coordenadas matemáticas. Para referirte a los vehículos involucrados, básate en el análisis visual: llámalos por su color (ej: "el auto rojo"), por su tipo ("el kart del interior", "el monoplaza"), por su número si es visible en los fotogramas, o en su defecto bajo los roles deportivos de "Auto Atacante" y "Auto Defensor". El lenguaje debe ser 100% humano, limpio y profesional.
- Traduce toda la lectura de movimiento a lenguaje deportivo humano; el espectador jamás debe ver términos técnicos.
- Resume el incidente en UN SOLO PÁRRAFO CORTO (2 a 4 oraciones) centrado en la aproximación, el vértice y el contacto. No listes líneas por fotograma individual ni fotogramas vacíos: el informe debe ser corto, preciso y directo al veredicto.
- La justificación reglamentaria debe tener como máximo 2 o 3 oraciones concisas y al grano.
- PROHIBIDO en el informe final: las palabras "conf" o "confianza", decimales o porcentajes técnicos (p. ej. "0.31"), la palabra "píxeles", listas de coordenadas como "[276,450,364,480]", y menciones a "YOLO", "modelo", "algoritmo" o "detección". El informe lo lee un aficionado: debe sonar a veredicto de comisario, nunca a registro de máquina.
- El campo "culpable" debe contener ÚNICAMENTE el rol o color limpio (ej: "El auto que iba por el interior", "El vehículo atacante"); jamás etiquetas técnicas ni porcentajes de confianza.

FORMATO DE RESPUESTA EXIGIDO (Devuelve estrictamente este JSON):
{
  "analisis_temporal": "Resumen compacto en un párrafo corto: aproximación, vértice y contacto (prohibido listar fotograma por fotograma o líneas vacías).",
  "culpable": "Solo el rol o color limpio del responsable (ej: 'El auto atacante', 'El auto rojo', 'El auto que iba por el interior').",
  "justificacion_reglamentaria": "El artículo del reglamento que se violó, en 2 o 3 oraciones concisas y en lenguaje deportivo.",
  "sancion_sugerida": "Advertencia / +5 Segundos / +10 Segundos / Drive Through / Incidente de carrera"
}

--- EJEMPLO FEW-SHOT (calibra formato, tono y brevedad; NUNCA copies sus datos) ---
Caso hipotético de entrada:
- Inicio (frenada): auto verde #33 por el interior y auto plateado #8 por fuera, alineados hacia el vértice.
- Tramo medio: el #33 no lleva su eje delantero por delante del retrovisor del #8 y frena tarde, quedándose pegado a él.
- Impacto: contacto en el vértice y ambos salen de la trazada.
Salida esperada (únicamente el JSON, sin texto adicional):
{
  "analisis_temporal": "El auto verde #33 atacó por el interior en la frenada, pero no completó la posición antes del vértice: frenó tarde y se mantuvo pegado al auto plateado #8 durante toda la aproximación. En el vértice se produjo el contacto y ambos perdieron la trazada.",
  "culpable": "El auto verde número 33 (atacante por el interior)",
  "justificacion_reglamentaria": "El atacante no superó con su eje delantero el retrovisor del defensor antes del vértice y bloqueó neumáticos. Es una colisión evitable y la culpa corresponde al auto interior (ARTÍCULO 1).",
  "sancion_sugerida": "+5 Segundos"
}
--- FIN DEL EJEMPLO ---
Aplica exactamente este mismo proceso de pensamiento, formato y nivel de detalle al INCIDENTE REAL que se te adjunta a continuación."""

# ==============================================================================
# 2. PROCESAMIENTO DE VIDEO + VISIÓN POR COMPUTADORA (YOLOv11)
# ==============================================================================

# --- 2.1 MOTOR DE DETECCIÓN GEOMÉTRICA (YOLOv11-n) ----------------------------

def _inicializar_modelo_yolo():
    """
    Carga YOLOv11-n al arrancar la app. El modelo ligero detecta vehículos de
    forma nativa en sus clases COCO 2 ('car') y 7 ('truck'), suficientes para
    autos de turismo, karts y trucks en pista.

    La descarga de 'yolo11n.pt' (~5 MB) ocurre una única vez en la primera
    ejecución. Si ultralytics no está instalado o la carga falla (p. ej. sin
    internet en el primer arranque), devuelve None y el pipeline continúa en
    modo solo-VLM (sin telemetría geométrica).
    """
    if YOLO is None:
        print(f"[YOLO] AVISO: ultralytics no disponible ({_ERROR_IMPORT_YOLO}). Telemetría desactivada.")
        return None
    try:
        modelo = YOLO(MODELO_YOLO)
        print(f"[YOLO] Modelo '{MODELO_YOLO}' cargado. Clases vehículo: {CLASES_VEHICULO}.")
        return modelo
    except Exception as exc:  # pragma: no cover - ruta defensiva
        print(f"[YOLO] AVISO: no se pudo cargar '{MODELO_YOLO}': {exc}")
        return None


yolo_model = _inicializar_modelo_yolo()


def _detectar_vehiculos(frame) -> List[Dict[str, object]]:
    """
    Ejecuta la inferencia YOLO sobre un fotograma y devuelve hasta los DOS
    vehículos principales (mayor confianza) con su caja [x1, y1, x2, y2] en
    píxeles del fotograma 640x480, su clase y su confianza.
    """
    if yolo_model is None:
        return []
    resultado = yolo_model(frame, verbose=False, conf=CONFIANZA_YOLO)[0]
    if resultado.boxes is None or len(resultado.boxes) == 0:
        return []

    cajas = resultado.boxes.xyxy.cpu().numpy()
    clases = resultado.boxes.cls.cpu().numpy().astype(int)
    confianzas = resultado.boxes.conf.cpu().numpy()

    detectados: List[Dict[str, object]] = []
    for caja, clase, confianza in zip(cajas, clases, confianzas):
        etiqueta = CLASES_VEHICULO.get(int(clase))
        if etiqueta is None:  # Ignora personas, señales y demás objetos
            continue
        detectados.append(
            {
                "caja": [float(v) for v in caja],
                "clase": etiqueta,
                "confianza": float(confianza),
            }
        )
    detectados.sort(key=lambda d: float(d["confianza"]), reverse=True)
    return detectados[:2]


def _distancia_centros_px(caja_a: List[float], caja_b: List[float]) -> float:
    """Distancia euclídea, en píxeles, entre los centros de las dos cajas."""
    ax = (caja_a[0] + caja_a[2]) / 2.0
    ay = (caja_a[1] + caja_a[3]) / 2.0
    bx = (caja_b[0] + caja_b[2]) / 2.0
    by = (caja_b[1] + caja_b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


def _hay_solapamiento(caja_a: List[float], caja_b: List[float]) -> bool:
    """True si las cajas se intersecan: señal geométrica de contacto."""
    return not (
        caja_a[2] < caja_b[0]
        or caja_b[2] < caja_a[0]
        or caja_a[3] < caja_b[1]
        or caja_b[3] < caja_a[1]
    )


def _separacion_cualitativa(distancia_px: float) -> str:
    """
    Traduce la distancia entre los centros de las cajas a una categoría humana
    (sin cifras ni unidades técnicas), lista para el lenguaje deportivo.
    """
    if distancia_px < 60:
        return "muy corta (al alcance de un toque)"
    if distancia_px < 140:
        return "corta (al alcance de un adelantamiento)"
    if distancia_px < 300:
        return "media"
    return "amplia (sin interacción inminente)"


def _telemetria_de_fotograma(
    numero: int, frame, distancia_previa: float | None
) -> tuple[str, float | None]:
    """
    Genera la línea de LECTURA DE MOVIMIENTO de UN fotograma, ya traducida a
    lenguaje deportivo y SIN ningún dato técnico copiable (ni confianzas, ni
    decimales, ni coordenadas, ni "píxeles"). Ejemplo de salida:
      "Fotograma 3: dos vehículos en pista. Separación relativa: corta
       (al alcance de un adelantamiento). Tendencia: se ACERCAN entre sí."

    Devuelve (linea, distancia_actual) para encadenar la tendencia del siguiente
    fotograma. Nunca lanza excepción: ante fallo de inferencia degrada el texto.
    """
    if yolo_model is None:
        return (
            f"Fotograma {numero}: sin lectura de movimiento disponible; usa solo la evidencia visual.",
            None,
        )

    try:
        detectados = _detectar_vehiculos(frame)
    except Exception:  # pragma: no cover - ruta defensiva
        return (
            f"Fotograma {numero}: la lectura automática falló; usa solo la evidencia visual.",
            None,
        )

    if len(detectados) < 2:
        if not detectados:
            return f"Fotograma {numero}: no se distinguen vehículos con claridad.", None
        return (
            f"Fotograma {numero}: solo un vehículo claramente visible; "
            "sin interacción medible entre dos autos.",
            None,
        )

    auto_1, auto_2 = detectados[0], detectados[1]
    distancia = _distancia_centros_px(auto_1["caja"], auto_2["caja"])

    if distancia_previa is None:
        tendencia = "primer registro de la secuencia"
    elif distancia - distancia_previa < -UMBRAL_TENDENCIA_PX:
        tendencia = "se ACERCAN entre sí"
    elif distancia - distancia_previa > UMBRAL_TENDENCIA_PX:
        tendencia = "se SEPARAN entre sí"
    else:
        tendencia = "mantienen la distancia (paralelos)"

    linea = (
        f"Fotograma {numero}: dos vehículos en pista. "
        f"Separación relativa: {_separacion_cualitativa(distancia)}. "
        f"Tendencia: {tendencia}."
    )
    if _hay_solapamiento(auto_1["caja"], auto_2["caja"]):
        linea += " ¡Los vehículos se solapan: contacto o toque inminente!"
    return linea, distancia


def _contar_frames_secuencial(cap: cv2.VideoCapture) -> int:
    """Cuenta frames con grab() cuando el contenedor no informa CAP_PROP_FRAME_COUNT."""
    total = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while total < MAX_INTENTOS_CONTEO and cap.grab():
        total += 1
    return total


def extraer_fotogramas_base64(
    ruta_video: str,
    max_frames: int = MAX_FOTOGRAMAS,
    ventana: tuple[float, float] | None = None,
) -> tuple[List[str], str]:
    """
    Extrae como máximo `max_frames` fotogramas EQUIDISTANTES dentro de la ventana
    temporal elegida por el usuario en el Trimmer del frontend (o de todo el
    clip si `ventana` es None o no es utilizable) y devuelve:

      * fotogramas: lista de cadenas Base64 (JPEG 640x480, calidad 85).
      * informe_telemetria_yolo: lectura de movimiento de YOLOv11, una línea por
        fotograma en lenguaje deportivo (sin cifras, coordenadas ni confianzas).

    Parámetros
    ----------
    ventana : tuple[float, float] | None
        (inicio_s, fin_s) con el segundo inicial y final de la maniobra. OpenCV
        salta directamente a frame_start = int(inicio_s * fps) y solo procesa
        fotogramas hasta frame_end = int(fin_s * fps): la IA recibe ÚNICAMENTE
        el recorte elegido, con la densidad concentrada donde está el toque.

    Optimización para GPU de 6 GB de VRAM:
      * cv2.resize OBLIGATORIO a 640x480 px por fotograma.
      * Compresión JPEG calidad 85 con cv2.imencode.
      * Conversión a Base64 recién cuando el JPEG ya está reducido.
      * YOLOv11-n corre sobre el MISMO fotograma 640x480 que verá el VLM, de
        modo que las coordenadas de la telemetría quedan en píxeles del VLM.

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

        # --- 2) Ventana del Trimmer -> límites de fotograma MATEMÁTICOS --------
        # `frame_start` y `frame_end` se calculan con TRUNCADO (int), tal como
        # exige el recorte estricto, y TODO el muestreo posterior vive SOLO
        # entre ambos índices: jamás se lee ni se envía a la IA un fotograma
        # fuera del rango elegido por el usuario.
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_start, frame_end = 0, total_frames - 1
        if ventana is not None:
            inicio_s, fin_s = ventana
            if fin_s > inicio_s:
                # Recorte ESTRICTO: sin fallback al clip completo aunque el
                # rango colisione con los bordes del archivo (se recorta a los
                # bordes). Sin FPS no hay matemática posible -> error claro.
                if fps <= 0:
                    raise ValueError(
                        "OpenCV no pudo determinar los FPS del video: "
                        "no es posible aplicar el recorte con precisión."
                    )
                frame_start = max(0, min(int(inicio_s * fps), total_frames - 1))
                frame_end = max(frame_start, min(int(fin_s * fps), total_frames - 1))
            # (fin <= inicio -> "sin recorte": clip completo; es además el
            # contrato legacy de /analizar con 0.0 / 0.0.)

        # Muestreo equidistante EXCLUSIVAMENTE dentro de [frame_start, frame_end]
        # (5 fotogramas para Ollama; 15-20 para Gemini, según `max_frames`).
        cantidad = min(max_frames, frame_end - frame_start + 1)
        indices = np.unique(
            np.linspace(frame_start, frame_end, cantidad).round().astype(int)
        )

        # --- 3) Extracción -> resize 640x480 -> YOLO -> JPEG 85 -> Base64 ------
        fotogramas: List[str] = []
        lineas_telemetria: List[str] = []
        distancia_previa: float | None = None

        # Salto directo al inicio del recorte: OpenCV posiciona el decoder en
        # `frame_start` (sin decodificar el tramo anterior) y desde AHÍ solo se
        # avanza secuencialmente hacia delante.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_start))
        objetivos = {int(indice) for indice in indices}
        contador = int(frame_start)

        while contador <= frame_end:
            # CORTE DURO: si el contador de frames del archivo supera
            # `frame_end`, la lectura se detiene por completo.
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > frame_end:
                break

            leido, imagen = cap.read()
            if not leido or imagen is None:
                break

            if contador in objetivos:
                # Solo el fotograma objetivo entra en la cadena: si el decoder
                # reporta una posición desviada, se re-sincroniza al índice exacto.
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != contador + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, contador)
                    leido, imagen = cap.read()
                    if not leido or imagen is None:
                        break

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
                    contador += 1  # avanza SIEMPRE: en un while, un `continue`
                    continue        # sin incremento dejaría el motor colgado

                # --- Telemetría YOLO del fotograma (misma numeración que el VLM) ---
                numero_fotograma = len(fotogramas) + 1
                linea, distancia_previa = _telemetria_de_fotograma(
                    numero_fotograma, imagen, distancia_previa
                )
                lineas_telemetria.append(linea)

                fotogramas.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
                if len(fotogramas) >= len(objetivos):
                    break  # rango ya cubierto: ahorra decodificar de más

            contador += 1

        if not fotogramas:
            raise ValueError("No se pudo extraer ningún fotograma del video.")
        return fotogramas, "\n".join(lineas_telemetria)
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


def _leer_reglamento() -> str:
    """
    RAG local: lee 'reglamento.txt' (artículos de maniobras) para inyectarlo
    como FUENTE 1 en el mensaje del usuario. Si el archivo falta o está vacío,
    devuelve un aviso de degradación sin detener el análisis.
    """
    ruta = DIRECTORIO_BASE / "reglamento.txt"
    try:
        texto = ruta.read_text(encoding="utf-8").strip()
    except OSError:
        return "No se pudo leer el reglamento local (reglamento.txt ausente)."
    return texto or "El reglamento local está vacío."


def _construir_mensaje_usuario(
    contexto: str | None,
    reglamento: str,
    telemetria: str,
    total_fotogramas: int = MAX_FOTOGRAMAS_LOCAL,
    ventana: tuple[float, float] | None = None,
) -> str:
    """
    Construye el User Message para el motor de análisis inyectando las TRES
    FUENTES DE VERDAD:
      1) REGLAMENTO OFICIAL    -> texto leído de 'reglamento.txt' (RAG local).
      2) LECTURA DE MOVIMIENTO -> resumen interno de YOLOv11 ya traducido a
         lenguaje deportivo (sin cifras, coordenadas ni confianzas: nada que
         pueda copiarse tal cual al informe).
      3) CONTEXTO DEL USUARIO  -> texto opcional de la casilla de la web (máx.
         MAX_CONTEXTO_CHARS caracteres).

    `total_fotogramas` indica cuántos fotogramas viajan adjuntos (5 en local,
    hasta 20 en nube) y se anuncian en orden cronológico dentro del mensaje.
    `ventana` (inicio_s, fin_s) declara el recorte estricto del Trimmer: si
    existe, el mensaje advierte a la IA de que SOLO existe información de
    ese fragmento y nada del resto del video.
    """
    texto = (contexto or "").strip()
    if len(texto) > MAX_CONTEXTO_CHARS:
        texto = texto[:MAX_CONTEXTO_CHARS]
    texto = texto.rstrip(" .") or "Ninguno aportado"

    # Aviso de recorte estricto (solo con ventana real): la IA debe saber que
    # las imágenes pertenecen ÚNICAMENTE a ese fragmento y a nada más.
    fragmento = ""
    if ventana is not None and ventana[1] > ventana[0]:
        inicio_s, fin_s = ventana
        fragmento = (
            "=== FRAGMENTO ANALIZADO (RECORTE ESTRICTO) ===\n"
            "Las imágenes adjuntas corresponden única y exclusivamente al fragmento de "
            f"tiempo delimitado entre el segundo {inicio_s:.3f} y el segundo {fin_s:.3f} "
            "del video original. No hay información del resto del video.\n\n"
        )

    return (
        f"Analiza el incidente mostrado en la secuencia adjunta ({total_fotogramas} "
        "fotogramas en orden cronológico). Dispones de TRES FUENTES DE VERDAD obligatorias:\n\n"
        "=== FUENTE 1 · REGLAMENTO OFICIAL (RAG) ===\n"
        f"{reglamento}\n\n"
        "=== FUENTE 2 · LECTURA DE MOVIMIENTO DE LOS VEHÍCULOS (uso interno: "
        "NO copies sus etiquetas ni sus frases; tradúcela a lenguaje deportivo) ===\n"
        f"{telemetria}\n\n"
        "=== FUENTE 3 · CONTEXTO APORTADO POR EL USUARIO ===\n"
        f"{texto}\n\n"
        f"{fragmento}"
        "INSTRUCCIONES FINALES: contrasta lo que VES en las imágenes con la FUENTE 2 "
        "(quién se acercaba a quién y en qué momento hubo solapamiento o contacto) "
        "y aplica el ARTÍCULO exacto de la FUENTE 1 que corresponda, citándolo en "
        "la justificación. Redacta el informe según la REGLA DE ESTILO DEL "
        "INFORME: lenguaje deportivo humano, sin términos técnicos, breve y al "
        "grano. Devuelve estrictamente la estructura JSON requerida."
    )


# --- Red de seguridad de formato -----------------------------------------------
# Aunque la REGLA DE ESTILO DEL INFORME lo prohíbe, por si la IA "arrastrara"
# telemetría cruda (p. ej. "Auto 1 (auto, conf 0.31)", "[276,450,364,480]" o
# "87 píxeles"), estas expresiones la eliminan ANTES de devolver el JSON.
_PATRONES_TECNICOS = (
    # Paréntesis técnicos completos: "(auto, conf 0.31)", "(auto)", "(truck/kart)"...
    (
        re.compile(
            r"\s*\(\s*(?:auto\s*[12]|truck(?:/kart)?|car|auto)?\s*,?\s*"
            r"(?:conf(?:ianza)?\s*[:=]?\s*[\d.,]+)?\s*\)",
            re.IGNORECASE,
        ),
        "",
    ),
    # Confianzas con cifra: "conf 0.86", "confianza: 0,31", "conf 31%"
    (re.compile(r"\bconf(?:ianza)?\s*[:=]?\s*\d+(?:[.,]\d+)?\s*%?", re.IGNORECASE), ""),
    # La palabra suelta "conf" o "confianza"
    (re.compile(r"\bconf(?:ianza)?\b", re.IGNORECASE), ""),
    # Decimales de confianza sueltos: "0.31" / "0,31" (protege versiones "2.0.0")
    (re.compile(r"(?<![.\d,])0[.,]\d+(?![\d])"), ""),
    # Coordenadas / cajas: "[276,450,364,480]"
    (re.compile(r"\[\s*\d+(?:[.,]\d+)?\s*(?:,\s*\d+(?:[.,]\d+)?\s*)*\]"), ""),
    # "87 píxeles", "12 px" -> expresión cualitativa
    (
        re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:píxeles|pixeles|px)\b", re.IGNORECASE),
        "una distancia relativa",
    ),
    (re.compile(r"\b(?:en|de)\s+píxeles\b", re.IGNORECASE), "en pista"),
    # Menciones al motor de visión
    (re.compile(r"\bYOLO(?:v?\d+)?\b", re.IGNORECASE), "el sistema de análisis"),
    (re.compile(r"\bmodelo\s+de\s+visión\b", re.IGNORECASE), "análisis"),
    # Limpieza final: paréntesis vacíos, espacios y puntuación huérfana
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"\s+([,.;:])"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
)


def _limpiar_jerga_tecnica(texto: str) -> str:
    """
    Red de seguridad final del informe: elimina cualquier resto de telemetría
    cruda que la IA pudiera arrastrar (confianzas, decimales, coordenadas,
    'píxeles', etiquetas YOLO...) y normaliza espacios y puntuación.
    """
    limpio = texto
    for patron, reemplazo in _PATRONES_TECNICOS:
        limpio = patron.sub(reemplazo, limpio)
    return limpio.strip()


# ==============================================================================
# 3.b MOTORES DE ANÁLISIS: OLLAMA (LOCAL) Y GEMINI (NUBE)
# ==============================================================================

def _quitar_cercas_markdown(texto: str) -> str:
    """Elimina las cercas ```json ... ``` si un modelo las añadiera al JSON."""
    limpio = texto.strip()
    if limpio.startswith("```"):
        limpio = limpio.split("\n", 1)[1] if "\n" in limpio else ""
        limpio = limpio.rstrip()
        if limpio.endswith("```"):
            limpio = limpio[:-3]
    return limpio.strip()


def _consultar_ollama(mensaje_usuario: str, imagenes: List[str]) -> str:
    """
    Motor LOCAL: consulta al modelo de visión de Ollama ('qwen2.5vl:7b') con las
    imágenes en orden cronológico y devuelve su texto crudo (JSON esperado).
    """
    try:
        respuesta = ollama.chat(
            model=MODELO_VLM,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje_usuario, "images": imagenes},
            ],
            format="json",           # Ollama fuerza salida JSON válida
            keep_alive="1h",         # Mantiene el modelo residente: evita la
                                     # recarga (~2-4 min) entre análisis seguidos
            options={
                "temperature": 0.2,  # Criterio "reglamentario" y repetible
                "num_predict": 900,  # Holgura para el informe
                # Presupuesto: ~5.300 tokens de imágenes + ~1.400 del prompt
                # CoT/Few-Shot + ~400 del reglamento (RAG) + lectura de
                # movimiento ≈ 7.100; con num_ctx 10240 queda margen holgado.
                # (El num_ctx por defecto de Ollama, 4096, daría error 400.)
                "num_ctx": 10240,
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
    return (respuesta.get("message", {}) or {}).get("content", "").strip()


def _consultar_gemini(mensaje_usuario: str, imagenes: List[str]) -> str:
    """
    Motor NUBE: consulta a Gemini (REST generateContent) con las imágenes
    adjuntas en orden cronológico. La clave se lee de GEMINI_API_KEY (variable
    de entorno o archivo '.env' local) y viaja en la cabecera 'x-goog-api-key'.

    MODO JSON STRICT: 'responseMimeType: application/json' + 'responseSchema'
    con las 4 claves obligatorias, de modo que Google NUNCA devuelva un string
    vacío, nulo o malformado: la API rechaza cualquier salida que no sea el
    objeto JSON exigido. Resiliente: reintenta 3 veces ante saturación temporal
    de Google (503) o fallos de red, con esperas crecientes de 5 s y 10 s.
    """
    if not CLAVE_GEMINI:
        raise HTTPException(
            status_code=503,
            detail=(
                "Falta la clave de Gemini. Crea un archivo '.env' en la raíz del "
                "proyecto con la línea GEMINI_API_KEY=tu_clave (ver '.env.example') "
                "o define la variable de entorno GEMINI_API_KEY."
            ),
        )

    partes: List[Dict[str, object]] = [{"text": mensaje_usuario}]
    for imagen_b64 in imagenes:
        partes.append({"inlineData": {"mimeType": "image/jpeg", "data": imagen_b64}})

    carga = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": partes}],
        "generationConfig": {
            "temperature": 0.2,                # Criterio reglamentario y repetible
            "responseMimeType": "application/json",   # JSON NATIVO de Google
            # Esquema nativo: Gemini NO puede devolver JSON roto ni sin claves.
            "responseSchema": {
                "type": "object",
                "properties": {
                    "analisis_temporal": {"type": "string"},
                    "culpable": {"type": "string"},
                    "justificacion_reglamentaria": {"type": "string"},
                    "sancion_sugerida": {"type": "string"},
                },
                "required": [
                    "analisis_temporal",
                    "culpable",
                    "justificacion_reglamentaria",
                    "sancion_sugerida",
                ],
            },
        },
    }
    peticion = urllib.request.Request(
        f"{URL_GEMINI}/{MODELO_GEMINI}:generateContent",
        data=json.dumps(carga).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": CLAVE_GEMINI,
        },
        method="POST",
    )
    # Reintentos ante saturación temporal de Google ("high demand") y fallos de
    # red: 3 intentos con espera creciente (5 s, 10 s). Los errores definitivos
    # (clave inválida, payload incorrecto) se reportan sin reintentar.
    intentos = 3
    espera_inicial = 5
    for intento in range(1, intentos + 1):
        try:
            with urllib.request.urlopen(peticion, timeout=300) as respuesta:
                datos_brutos = respuesta.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            codigo = exc.code
            detalle = exc.read().decode("utf-8", errors="replace")[:400]
            transitorio = codigo in {408, 429, 500, 502, 503, 504}
            if not transitorio:
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini rechazó la petición (HTTP {codigo}): {detalle}",
                ) from exc
            if intento == intentos:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Gemini está saturado o falló de forma temporal (HTTP {codigo}) tras "
                        f"{intento} intentos. Detalle: {detalle}. Reintenta en unos segundos "
                        "o usa el motor Local (Ollama)."
                    ),
                ) from exc
            time.sleep(espera_inicial * (2 ** (intento - 1)))
        except Exception as exc:  # URLError, timeout, DNS...
            if intento == intentos:
                raise HTTPException(
                    status_code=503,
                    detail=f"No se pudo contactar con la API de Gemini. Detalle: {exc}",
                ) from exc
            time.sleep(espera_inicial * (2 ** (intento - 1)))
    else:  # pragma: no cover - el bucle siempre termina con break o excepción
        raise HTTPException(status_code=502, detail="Gemini no devolvió respuesta utilizable.")

    try:
        cuerpo = json.loads(datos_brutos)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini devolvió una respuesta que no es JSON válido.",
        ) from exc

    candidatos = cuerpo.get("candidates") or []
    if not candidatos:
        bloqueo = (cuerpo.get("promptFeedback") or {}).get("blockReason")
        raise HTTPException(
            status_code=502,
            detail=f"Gemini no devolvió ninguna respuesta ({bloqueo or 'sin candidatos'}).",
        )
    partes_respuesta = (candidatos[0].get("content") or {}).get("parts") or []
    texto = "".join(str(parte.get("text", "")) for parte in partes_respuesta).strip()
    if not texto:
        raise HTTPException(status_code=502, detail="Gemini devolvió una respuesta vacía.")
    return texto


def _consultar_motor(motor: str, mensaje_usuario: str, imagenes: List[str]) -> str:
    """Enruta la consulta al motor elegido en la interfaz ('local' | 'nube')."""
    if motor == MOTOR_NUBE:
        return _consultar_gemini(mensaje_usuario, imagenes)
    return _consultar_ollama(mensaje_usuario, imagenes)


def _informe_por_defecto(motor: str = MOTOR_LOCAL) -> Dict[str, str]:
    """
    Informe SEGURO de contingencia: si la IA falla, está saturada o devuelve
    un string vacío, nulo o malformado, el frontend SIEMPRE recibe un JSON
    válido con el contrato de claves que 'app.js' sabe renderizar (nunca se
    queda colgado esperando). Para la nube se usa el mensaje de contingencia
    que pide acortar el rango; en local, el genérico de reintento.
    """
    if motor == MOTOR_NUBE:
        analisis = "Error al procesar el video en la nube. Intente acortar el rango."
        culpable = "No determinado"
        justificacion = "N/A"
        sancion = "N/A"
    else:
        analisis = "La IA no pudo procesar este clip específico, intente nuevamente."
        culpable = "Error de procesamiento"
        justificacion = (
            "Sin veredicto posible: el análisis automático no pudo completarse. "
            "Reintenta con el clip completo o prueba con el otro motor."
        )
        sancion = "N/A"
    return {
        "analisis_temporal": analisis,
        "culpable": culpable,
        "justificacion_reglamentaria": justificacion,
        "sancion_sugerida": sancion,
        "analisis_tecnico": (
            f"ANÁLISIS TEMPORAL:\n{analisis}\n\nJUSTIFICACIÓN REGLAMENTARIA:\n{justificacion}"
        ),
    }


def generar_informe_json(
    fotogramas_base64: List[str],
    contexto: str | None = None,
    informe_telemetria_yolo: str = "",
    motor: str = MOTOR_LOCAL,
    ventana: tuple[float, float] | None = None,
) -> Dict[str, str]:
    """
    Envía la secuencia de fotogramas Base64 al motor elegido (Ollama local o
    Gemini en la nube) aplicando Few-Shot + Chain of Thought (SYSTEM_PROMPT) y
    devuelve el informe estructurado con las claves:
      * analisis_temporal           -> resumen compacto (aproximación/vértice/contacto)
      * culpable                    -> identificación limpia del vehículo culpable
      * justificacion_reglamentaria -> artículo del reglamento violado
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
    informe_telemetria_yolo : str
        Lectura de movimiento de YOLOv11 (una línea por fotograma, ya traducida a
        lenguaje deportivo) que se inyecta como Fuente 2 del User Message; el
        reglamento local ('reglamento.txt', RAG) entra como Fuente 1 y el
        contexto del usuario como Fuente 3.
    motor : str
        "local" (Ollama, por defecto) o "nube" (Gemini). La limpieza de formato
        del informe se aplica por igual a ambos motores.

    Ante cualquier fallo de la IA (conexión, saturación, clave ausente, JSON
    vacío o roto) devuelve un informe SEGURO por defecto; solo lanza
    HTTPException si no hay fotogramas que analizar.
    """
    if not fotogramas_base64:
        raise HTTPException(status_code=400, detail="No hay fotogramas para analizar.")

    # --- User Message: reglamento (RAG) + lectura de movimiento + contexto ----
    reglamento = _leer_reglamento()  # RAG local: se relee en cada petición
    mensaje_usuario = _construir_mensaje_usuario(
        contexto,
        reglamento,
        informe_telemetria_yolo,
        len(fotogramas_base64),
        ventana=ventana,
    )

    # --- Consulta PROTEGIDA al motor (Ollama local o Gemini en la nube) -------
    # Try/Except ESTRICTO: cualquier fallo (conexión, saturación, clave,
    # string vacío, nulo o malformado) se intercepta y devuelve OBLIGATORIAMENTE
    # el JSON estructurado de contingencia. Gracias al responseMimeType +
    # responseSchema, Gemini NUNCA entrega respuestas vacías o rotas: si algo
    # saliera mal igual, este bloque garantiza que el frontend siempre reciba
    # datos y nunca se quede colgado esperando.
    # NOTA DE ARQUITECTURA: no se usa el SDK 'google-generativeai' (que exigiría
    # 'generation_config={"response_mime_type": "application/json"}' como pide el
    # enunciado) porque el proyecto ya habla con Gemini vía REST directo (sin
    # dependencias nuevas): 'responseMimeType' + 'responseSchema' viajan en
    # 'generationConfig' del payload JSON, con idéntico efecto STRICT.
    try:
        contenido = _quitar_cercas_markdown(
            _consultar_motor(motor, mensaje_usuario, fotogramas_base64)
        )
        # El motor puede devolver None (no solo ""): se trata como vacío
        # ANTES de json.loads para que caiga en la contingencia.
        if contenido is None or not str(contenido).strip():
            raise ValueError("la respuesta de la IA llegó vacía o nula")
        datos = json.loads(contenido)
        if not isinstance(datos, dict):
            raise ValueError("la respuesta de la IA no es un objeto JSON")
    except HTTPException as exc:
        print(f"[IA] Fallo del motor '{motor}': {getattr(exc, 'detail', exc)}")
        return _informe_por_defecto(motor)
    except Exception as exc:
        print(f"[IA] Respuesta inservible del motor '{motor}': {exc}")
        return _informe_por_defecto(motor)

    # Un {} válido pero sin contenido útil también dispara la contingencia:
    # se exige que las CUATRO claves tengan texto no vacío.
    claves_requeridas = (
        "analisis_temporal",
        "culpable",
        "justificacion_reglamentaria",
        "sancion_sugerida",
    )
    if not datos or not all(str(datos.get(clave) or "").strip() for clave in claves_requeridas):
        print(f"[IA] El motor '{motor}' devolvió un JSON vacío o incompleto.")
        return _informe_por_defecto(motor)

    # --- Nuevas claves del prompt CoT + clave histórica para app.js ------------
    # Se pasa la red de seguridad de formato: nada de telemetría cruda al usuario.
    analisis_temporal = _limpiar_jerga_tecnica(str(datos.get("analisis_temporal") or "").strip())
    justificacion = _limpiar_jerga_tecnica(
        str(datos.get("justificacion_reglamentaria") or "").strip()
    )
    culpable = _limpiar_jerga_tecnica(str(datos.get("culpable") or "").strip()) or "No determinado"
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
        analisis_tecnico = (
            _limpiar_jerga_tecnica(str(datos.get("analisis_tecnico") or "").strip())
            or "No disponible."
        )

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
    description="Backend local para análisis de incidentes en pista con IA de visión + YOLOv11 + RAG (Ollama local o Gemini en la nube).",
    version=VERSION_APP,
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


@app.get("/logo.png", include_in_schema=False)
def servir_logo() -> FileResponse:
    """Sirve el emblema CDA-ACA (favicon, cabecera, spinner y pie)."""
    return FileResponse(DIRECTORIO_BASE / "logo.png", media_type="image/png")


@app.get("/version", include_in_schema=False)
def obtener_version() -> JSONResponse:
    """
    Metadatos del desarrollo para el sello del pie de página: versión, fecha y
    capacidades activas (modelo VLM, pesos YOLO cargados y reglamento RAG).
    """
    return JSONResponse(
        content={
            "nombre": "Race Control",
            "version": VERSION_APP,
            "fecha": FECHA_ACTUALIZACION,
            "modelo_vlm": MODELO_VLM,
            "modelo_gemini": MODELO_GEMINI,
            "modelo_yolo": MODELO_YOLO if yolo_model is not None else None,
            "rag": (DIRECTORIO_BASE / "reglamento.txt").is_file(),
            "motores": sorted(MOTORES_VALIDOS),
        }
    )


@app.post("/analizar")
async def analizar_incidente(
    video: UploadFile = File(..., description="Clip del incidente (MP4/MOV; 15 MB en local, 100 MB en nube)"),
    contexto: str | None = Form(
        None,
        description="Contexto opcional de la maniobra (atacante/defensor, números/colores)",
    ),
    tiempo_inicio: float = Form(
        0.0,
        description="Segundo inicial del recorte elegido en el Trimmer (0 = clip completo)",
    ),
    tiempo_fin: float = Form(
        0.0,
        description="Segundo final del recorte elegido en el Trimmer (0 = clip completo)",
    ),
    motor: str = Form(
        MOTOR_LOCAL,
        description="Motor de análisis: 'local' (Ollama, máx. 5 fotogramas) o 'nube' (Gemini, hasta 20)",
    ),
) -> JSONResponse:
    """
    Pipeline completo: valida el clip -> extrae fotogramas con densidad según el
    motor elegido -> consulta a Ollama (local) o Gemini (nube) -> devuelve JSON.
    """
    # (0) Motor y límites dinámicos (densidad de fotogramas y peso del clip)
    motor_normalizado = (motor or MOTOR_LOCAL).strip().lower()
    if motor_normalizado not in MOTORES_VALIDOS:
        raise HTTPException(status_code=400, detail="Motor no válido. Usa 'local' o 'nube'.")
    if motor_normalizado == MOTOR_NUBE:
        max_fotogramas, tamano_maximo_mb = MAX_FOTOGRAMAS_NUBE, TAMANO_MAXIMO_MB_NUBE
    else:
        max_fotogramas, tamano_maximo_mb = MAX_FOTOGRAMAS_LOCAL, TAMANO_MAXIMO_MB_LOCAL
    tamano_maximo_bytes = tamano_maximo_mb * 1024 * 1024

    # (1) Formato del archivo
    extension = Path(video.filename or "").suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(status_code=400, detail="Formato no admitido. Envía un archivo .mp4 o .mov.")

    # (2) Peso máximo según motor (lectura acotada: no se carga nada más en memoria)
    datos = await video.read(tamano_maximo_bytes + 1)
    if not datos:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(datos) > tamano_maximo_bytes:
        if motor_normalizado == MOTOR_NUBE:
            detalle = (
                f"El clip supera los {tamano_maximo_mb} MB del motor Nube. "
                "Prueba con un recorte más liviano o reduce la resolución del video."
            )
        else:
            detalle = (
                f"El clip supera los {tamano_maximo_mb} MB del motor Local. "
                "Recorta un clip de 5 a 10 segundos o cambia al motor Nube "
                "(Gemini), que admite videos de 20 a 45 segundos."
            )
        raise HTTPException(status_code=413, detail=detalle)

    # (3) Extracción de fotogramas optimizados (densidad según motor) + YOLO
    # Sanidad del rango del Trimmer: sin tiempos negativos ni invertidos. El
    # recorte estricto exige inicio <= fin; 0.0 / 0.0 sigue significando
    # "clip completo" (contrato legacy documentado en los Form del endpoint).
    if tiempo_inicio < 0.0 or tiempo_fin < 0.0:
        raise HTTPException(
            status_code=400,
            detail="El rango del Trimmer no puede contener tiempos negativos.",
        )
    if tiempo_fin < tiempo_inicio:
        raise HTTPException(
            status_code=400,
            detail="El segundo final del recorte no puede ser anterior al inicial.",
        )

    ruta_temporal = guardar_clip_temporal(datos, extension)
    try:
        # Ventana del Trimmer: si el usuario no recortó, None = clip completo.
        ventana = (tiempo_inicio, tiempo_fin) if tiempo_fin > tiempo_inicio + 0.05 else None
        fotogramas, informe_telemetria_yolo = extraer_fotogramas_base64(
            ruta_temporal, max_fotogramas, ventana
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(ruta_temporal)

    # (4) Análisis con el motor elegido y (5) respuesta JSON directa al frontend
    # `ventana` viaja también al prompt: la IA es advertida de que SOLO existe
    # información del fragmento recortado (nada del resto del video).
    return JSONResponse(
        content=generar_informe_json(
            fotogramas,
            contexto,
            informe_telemetria_yolo,
            motor_normalizado,
            ventana=ventana,
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
