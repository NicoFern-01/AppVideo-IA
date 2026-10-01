"""
_test_validacion.py — Pruebas de la arquitectura híbrida (YOLOv11 + RAG + VLM).

Valida 'main.py' sin usar la web ni Ollama real, en 5 bloques:
  1) Lectura del reglamento (RAG local).
  2) Composición del User Message con las TRES fuentes de verdad.
  3) Telemetría YOLO unitaria (detección simulada): formato, tendencia y solape.
  4) YOLO real sobre un video de tráfico (_test_vehiculos.mp4): 5 fotogramas.
  5) Pipeline completo con ollama.chat simulado: imágenes + prompt + opciones.

Uso: python _test_validacion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# La consola de Windows puede no ser UTF-8: evita UnicodeEncodeError al imprimir.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIRECTORIO = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORIO))

import main  # noqa: E402  (importa ultralytics/YOLO y carga yolo11n.pt)

APROBADOS: list[str] = []
FALLOS: list[str] = []


def comprobar(condicion: bool, descripcion: str) -> None:
    if condicion:
        APROBADOS.append(descripcion)
        print(f"  [OK]    {descripcion}")
    else:
        FALLOS.append(descripcion)
        print(f"  [FALLO] {descripcion}")


def caja(x1, y1, x2, y2, conf=0.9, clase="auto"):
    return {"caja": [float(x1), float(y1), float(x2), float(y2)], "clase": clase, "confianza": conf}


# ---------------------------------------------------------------- 1) REGLAMENTO
print("\n=== 1) RAG: lectura de reglamento.txt ===")
reglamento = main._leer_reglamento()
primer_linea = reglamento.splitlines()[0].strip() if reglamento.splitlines() else ""
print(f"  (Primera línea del reglamento: {primer_linea!r})")
comprobar("ARTÍCULO" in reglamento.upper(), "El reglamento usa artículos numerados (ARTÍCULO ...)")
comprobar("Divebomb" in reglamento, "El reglamento menciona 'Divebomb'")
comprobar("frenado" in reglamento.lower(), "El reglamento cubre la zona de frenado")

# ------------------------------------------------ 2) MENSAJE CON LAS 3 FUENTES
print("\n=== 2) User Message: tres fuentes de verdad ===")
mensaje = main._construir_mensaje_usuario("Auto rojo atacante por dentro", reglamento, "Frame 1: prueba")
comprobar("FUENTE 1 · REGLAMENTO OFICIAL (RAG)" in mensaje, "Incluye la FUENTE 1 (reglamento)")
comprobar("FUENTE 2 · LECTURA DE MOVIMIENTO DE LOS VEHÍCULOS" in mensaje, "Incluye la FUENTE 2 (lectura de movimiento)")
comprobar("NO copies sus etiquetas" in mensaje, "La FUENTE 2 se marca como uso interno (no copiable)")
comprobar("FUENTE 3 · CONTEXTO APORTADO POR EL USUARIO" in mensaje, "Incluye la FUENTE 3 (contexto)")
comprobar(bool(primer_linea) and primer_linea in mensaje, "El reglamento completo viaja dentro del mensaje")
comprobar("Auto rojo atacante por dentro" in mensaje, "El contexto del usuario viaja dentro")
comprobar("Frame 1: prueba" in mensaje, "La telemetría viaja dentro")
mensaje_nube = main._construir_mensaje_usuario("ctx", reglamento, "tel", total_fotogramas=20)
comprobar("(20 fotogramas" in mensaje_nube, "El mensaje anuncia la densidad del motor (20 fotogramas en nube)")

# Recorte estricto: el prompt declara EXACTAMENTE el fragmento analizado.
mensaje_ventana = main._construir_mensaje_usuario("ctx", reglamento, "tel", ventana=(3.0, 7.5))
comprobar(
    "única y exclusivamente" in mensaje_ventana
    and "segundo 3.000" in mensaje_ventana
    and "segundo 7.500" in mensaje_ventana,
    "El mensaje declara el fragmento estricto (segundo 3.000 al 7.500)",
)
mensaje_sin_ventana = main._construir_mensaje_usuario("ctx", reglamento, "tel")
comprobar(
    "única y exclusivamente" not in mensaje_sin_ventana,
    "Sin ventana NO se inyecta la línea de fragmento",
)

# -------------------------------------- 3) TELEMETRÍA UNITARIA (DETECCIÓN FALSA)
print("\n=== 3) Lectura de movimiento unitaria (detección simulada) ===")
original_detectar = main._detectar_vehiculos

detecciones_lejos = [caja(100, 200, 200, 300, 0.95), caja(400, 200, 500, 300, 0.90)]  # separación amplia
detecciones_cerca = [caja(100, 200, 200, 300, 0.95), caja(300, 200, 400, 300, 0.90)]  # separación media
detecciones_solape = [caja(100, 200, 300, 300, 0.95), caja(250, 210, 450, 310, 0.90)]  # cajas intersectadas

try:
    main._detectar_vehiculos = lambda frame: detecciones_lejos
    linea_1, dist_1 = main._telemetria_de_fotograma(1, None, None)
    comprobar(linea_1.startswith("Fotograma 1: dos vehículos"), "El texto empieza con 'Fotograma 1: dos vehículos'")
    comprobar("primer registro" in linea_1, "Primer fotograma: 'primer registro de la secuencia'")

    main._detectar_vehiculos = lambda frame: detecciones_cerca
    linea_2, dist_2 = main._telemetria_de_fotograma(2, None, dist_1)
    comprobar("se ACERCAN entre sí" in linea_2, "300 -> 200 se describe como 'se ACERCAN entre sí'")

    linea_3, dist_3 = main._telemetria_de_fotograma(3, None, dist_2)
    comprobar("mantienen la distancia" in linea_3, "200 -> 200 se describe como 'mantienen la distancia'")

    main._detectar_vehiculos = lambda frame: detecciones_lejos
    linea_4, _ = main._telemetria_de_fotograma(4, None, dist_3)
    comprobar("se SEPARAN entre sí" in linea_4, "200 -> 300 se describe como 'se SEPARAN entre sí'")

    main._detectar_vehiculos = lambda frame: detecciones_solape
    linea_5, _ = main._telemetria_de_fotograma(5, None, dist_1)
    comprobar("se solapan" in linea_5.lower(), "Cajas intersectadas: aviso de contacto inminente")

    main._detectar_vehiculos = lambda frame: [detecciones_lejos[0]]
    linea_sola, dist_sola = main._telemetria_de_fotograma(6, None, None)
    comprobar("solo un vehículo" in linea_sola and dist_sola is None, "Un solo vehículo: lectura no concluyente")

    todas = " ".join([linea_1, linea_2, linea_3, linea_4, linea_5, linea_sola])
    comprobar("conf" not in todas.lower(), "La lectura de movimiento no contiene 'conf'")
    comprobar("píxeles" not in todas.lower(), "La lectura de movimiento no contiene 'píxeles'")
    comprobar("[" not in todas and "]" not in todas, "La lectura de movimiento no contiene coordenadas")
    comprobar("0." not in todas and "0," not in todas, "La lectura de movimiento no contiene decimales técnicos")
finally:
    main._detectar_vehiculos = original_detectar

# ---------------------------------------- 4) YOLO REAL SOBRE VIDEO DE TRÁFICO
print("\n=== 4) YOLO real sobre video de tráfico (_test_vehiculos.mp4) ===")
ruta_video = DIRECTORIO / "_test_vehiculos.mp4"
if main.yolo_model is None:
    comprobar(False, "El modelo YOLO no está cargado (ultralytics no disponible)")
elif not ruta_video.is_file():
    comprobar(False, "No existe _test_vehiculos.mp4 para la prueba real")
else:
    fotogramas, telemetria = main.extraer_fotogramas_base64(str(ruta_video), main.MAX_FOTOGRAMAS)
    comprobar(len(fotogramas) == 5, f"Se extrajeron 5 fotogramas (obtenidos: {len(fotogramas)})")
    lineas = [linea for linea in telemetria.splitlines() if linea.strip()]
    comprobar(len(lineas) == 5, f"La telemetría tiene 5 líneas (obtenidas: {len(lineas)})")
    comprobar(
        any("dos vehículos en pista" in linea for linea in lineas),
        "Al menos un fotograma con 2 vehículos medidos (lectura real)",
    )
    print("  --- Lectura de movimiento real generada por YOLO ---")
    for linea in lineas:
        print("    " + linea)
    todas_reales = " ".join(lineas).lower()
    comprobar("conf" not in todas_reales, "Lectura real sin 'conf'")
    comprobar("píxeles" not in todas_reales, "Lectura real sin 'píxeles'")
    comprobar("[" not in todas_reales, "Lectura real sin coordenadas")

    # --- Densidad NUBE: el mismo video, hasta 20 fotogramas equidistantes ----
    fotogramas_nube, telemetria_nube = main.extraer_fotogramas_base64(
        str(ruta_video), main.MAX_FOTOGRAMAS_NUBE
    )
    comprobar(
        len(fotogramas_nube) == 20,
        f"Motor nube: 20 fotogramas extraídos (obtenidos: {len(fotogramas_nube)})",
    )
    comprobar(
        len([l for l in telemetria_nube.splitlines() if l.strip()]) == 20,
        "Motor nube: 20 líneas de lectura de movimiento",
    )

# ------------------------------------- 5) PIPELINE COMPLETO CON OLLAMA SIMULADO
print("\n=== 5) Pipeline completo con ollama.chat simulado ===")
capturado: dict = {}


def chat_simulado(**kwargs):
    capturado.update(kwargs)
    return {
        "message": {
            "content": json.dumps(
                {
                    # Respuesta "SUCIA" a propósito: reproduce la fuga de datos
                    # crudos de YOLO que la red de seguridad debe limpiar.
                    "analisis_temporal": (
                        "Fotograma 1: Auto 1 (auto, conf 0.31) en [276,450,364,480] cerca del rival. "
                        "Fotograma 2: distancia de 12 píxeles entre ambos."
                    ),
                    "culpable": "Auto 1 (auto, conf 0.31)",
                    "justificacion_reglamentaria": (
                        "El vehículo YOLO detectado bloqueó neumáticos; colisión evitable (ARTÍCULO 1)."
                    ),
                    "sancion_sugerida": "+5 Segundos",
                }
            )
        }
    }


chat_original = main.ollama.chat
main.ollama.chat = chat_simulado
try:
    informe = main.generar_informe_json(
        ["QUJD"], "contexto de prueba", "Frame 1: telemetría de prueba"
    )
finally:
    main.ollama.chat = chat_original

comprobar(capturado.get("format") == "json", "Ollama recibe format='json'")
comprobar(capturado.get("keep_alive") == "1h", "Ollama recibe keep_alive='1h'")
opciones = capturado.get("options", {})
comprobar(
    opciones.get("num_ctx") == 10240,
    f"num_ctx ampliado a 10240 (recibido: {opciones.get('num_ctx')})",
)

mensajes = capturado.get("messages", [])
comprobar(len(mensajes) == 2, f"Hay 2 mensajes: system + user (recibidos: {len(mensajes)})")
if len(mensajes) == 2:
    comprobar("TRES FUENTES DE VERDAD" in mensajes[0]["content"], "El System Prompt exige las 3 fuentes")
    contenido_usuario = mensajes[1]["content"]
    comprobar(
        "FUENTE 1 · REGLAMENTO OFICIAL (RAG)" in contenido_usuario and primer_linea in contenido_usuario,
        "El reglamento llega en la FUENTE 1",
    )
    comprobar("Frame 1: telemetría de prueba" in contenido_usuario, "La telemetría YOLO llega en la FUENTE 2")
    comprobar("contexto de prueba" in contenido_usuario, "El contexto del usuario llega en la FUENTE 3")
    comprobar(mensajes[1].get("images") == ["QUJD"], "Las imágenes Base64 viajan en el mensaje")

comprobar(
    informe.get("sancion_sugerida") == "+5s",
    f"La sanción '+5 Segundos' se normaliza a '+5s' (recibido: {informe.get('sancion_sugerida')})",
)
comprobar(
    informe.get("analisis_tecnico", "").startswith("ANÁLISIS TEMPORAL:"),
    "La clave 'analisis_tecnico' (contrato con app.js) se compone con el análisis temporal",
)
comprobar(
    "ARTÍCULO 1" in informe.get("justificacion_reglamentaria", ""),
    "La justificación reglamentaria llega al JSON final",
)

# --- RED DE SEGURIDAD DE FORMATO: la respuesta "sucia" debe salir limpia -----
conjunto = json.dumps(informe, ensure_ascii=False).lower()
comprobar("conf" not in conjunto, "El informe final no contiene 'conf'")
comprobar("0.31" not in conjunto and "0,31" not in conjunto, "El informe final no contiene decimales de confianza")
comprobar("píxeles" not in conjunto, "El informe final no contiene 'píxeles'")
comprobar("[276" not in conjunto and "364,480" not in conjunto, "El informe final no contiene coordenadas")
comprobar("yolo" not in conjunto, "El informe final no menciona 'YOLO'")
comprobar(
    informe.get("culpable") == "Auto 1",
    f"El campo 'culpable' queda solo con el rol limpio (recibido: {informe.get('culpable')!r})",
)
for clave in ("analisis_temporal", "culpable", "justificacion_reglamentaria", "sancion_sugerida", "analisis_tecnico"):
    comprobar(clave in informe, f"El JSON final incluye '{clave}' (contrato con app.js)")

# ---------------- 6) MOTOR NUBE (GEMINI) CON RESPUESTA SIMULADA --------------
print("\n=== 6) Dispatcher de motores y motor Nube (Gemini simulado) ===")
comprobar(main.MAX_FOTOGRAMAS_LOCAL == 5, "Densidad local = 5 fotogramas fijos")
comprobar(main.MAX_FOTOGRAMAS_NUBE == 20, "Densidad nube = 20 fotogramas")
comprobar(main.MOTORES_VALIDOS == {"local", "nube"}, "Motores válidos: 'local' y 'nube'")
comprobar(bool(main.CLAVE_GEMINI), "Clave Gemini disponible (variable de entorno o .env local)")
comprobar(
    main._quitar_cercas_markdown('```json\n{"a": 1}\n```') == '{"a": 1}',
    "Las cercas markdown del JSON se eliminan",
)

capturado_nube: dict = {}


def gemini_simulado(mensaje_usuario, imagenes):
    capturado_nube["mensaje"] = mensaje_usuario
    capturado_nube["imagenes"] = list(imagenes)
    return json.dumps(
        {
            "analisis_temporal": "El auto rojo cerró la puerta al auto azul en la frenada y hubo contacto leve en el vértice.",
            "culpable": "El auto rojo (defensor)",
            "justificacion_reglamentaria": "El defensor cambió de trayectoria en zona de frenado y provocó el toque (ARTÍCULO 3).",
            "sancion_sugerida": "Advertencia",
        }
    )


gemini_original = main._consultar_gemini
main._consultar_gemini = gemini_simulado
try:
    informe_nube = main.generar_informe_json(
        ["QUJD", "RUZH"], "defensor rojo", "Fotograma 1: se ACERCAN entre sí.", motor="nube"
    )
finally:
    main._consultar_gemini = gemini_original

comprobar(
    len(capturado_nube.get("imagenes", [])) == 2,
    "El motor Nube recibe las imágenes en orden (2 adjuntas)",
)
comprobar(
    "(2 fotogramas" in capturado_nube.get("mensaje", ""),
    "El mensaje anuncia los fotogramas en orden cronológico",
)
comprobar(
    "FUENTE 1 · REGLAMENTO OFICIAL (RAG)" in capturado_nube.get("mensaje", ""),
    "El reglamento (RAG) también viaja al motor Nube",
)
comprobar(
    informe_nube.get("culpable") == "El auto rojo (defensor)",
    f"Respuesta Nube procesada con contrato intacto (culpable: {informe_nube.get('culpable')!r})",
)
comprobar(
    "ARTÍCULO 3" in informe_nube.get("justificacion_reglamentaria", ""),
    "La cita del artículo llega intacta desde la nube",
)
for clave in ("analisis_temporal", "culpable", "justificacion_reglamentaria", "sancion_sugerida", "analisis_tecnico"):
    comprobar(clave in informe_nube, f"El JSON de la nube incluye '{clave}' (contrato con app.js)")

# -------------- 7) TRIMMER (VENTANA TEMPORAL) Y FALLBACK SEGURO --------------
print("\n=== 7) Trimmer (ventana temporal) y fallback seguro ===")
ruta_test = DIRECTORIO / "_test_vehiculos.mp4"
if main.yolo_model is None or not ruta_test.is_file():
    comprobar(False, "No se pudo probar el Trimmer (falta YOLO o el video de prueba)")
else:
    import base64 as _b64
    import cv2 as _cv2
    import numpy as _np

    # (a) Ventana de 1 s (25 fps -> ~26 frames) pidiendo 5: todo dentro del rango.
    frames_ventana, tel_ventana = main.extraer_fotogramas_base64(
        str(ruta_test), 5, ventana=(2.0, 3.0)
    )
    comprobar(
        len(frames_ventana) == 5,
        f"Trimmer: 5 fotogramas dentro de la ventana (obtenidos: {len(frames_ventana)})",
    )
    comprobar(
        len([l for l in tel_ventana.splitlines() if l.strip()]) == 5,
        "Trimmer: 5 líneas de lectura de movimiento",
    )

    # (b) Ventana muy estrecha: menos frames disponibles que los pedidos.
    frames_angosta, _ = main.extraer_fotogramas_base64(
        str(ruta_test), 5, ventana=(2.0, 2.08)
    )
    comprobar(
        0 < len(frames_angosta) < 5,
        f"Trimmer: ventana estrecha acota los fotogramas (obtenidos: {len(frames_angosta)})",
    )

    # (c) El primer fotograma debe ser el del inicio del recorte (2.0 s).
    img = _cv2.imdecode(
        _np.frombuffer(_b64.b64decode(frames_ventana[0]), _np.uint8), _cv2.IMREAD_COLOR
    )
    cap = _cv2.VideoCapture(str(ruta_test))
    fps_test = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    cap.set(_cv2.CAP_PROP_POS_FRAMES, int(round(2.0 * fps_test)))
    leido, referencia = cap.read()
    cap.release()
    if leido and referencia is not None:
        # Mismo resize que el backend (INTER_AREA). El diff no es 0 por la pérdida
        # del JPEG q85, así que se compara además con un frame lejano: si el
        # recorte funciona, el frame correcto debe ser MUCHO más parecido.
        referencia = _cv2.resize(
            referencia,
            (main.ANCHO_FOTOGRAMA, main.ALTO_FOTOGRAMA),
            interpolation=_cv2.INTER_AREA,
        )
        cap_lejos = _cv2.VideoCapture(str(ruta_test))
        cap_lejos.set(_cv2.CAP_PROP_POS_FRAMES, int(round(7.0 * fps_test)))
        leido_lejos, frame_lejos = cap_lejos.read()
        cap_lejos.release()
        if leido_lejos and frame_lejos is not None:
            frame_lejos = _cv2.resize(
                frame_lejos,
                (main.ANCHO_FOTOGRAMA, main.ALTO_FOTOGRAMA),
                interpolation=_cv2.INTER_AREA,
            )
            diff_lejos = float(
                _np.mean(_np.abs(img.astype("int16") - frame_lejos.astype("int16")))
            )
        else:
            diff_lejos = 999.0

        diferencia = float(
            _np.mean(_np.abs(img.astype("int16") - referencia.astype("int16")))
        )
        comprobar(
            diferencia < 6.0 and diff_lejos > diferencia * 1.5,
            f"El 1er fotograma es el del inicio 2.0 s (diff={diferencia:.2f} vs "
            f"frame lejano 7.0 s={diff_lejos:.2f})",
        )
    else:
        comprobar(False, "No se pudo leer el frame de referencia para comparar")

    # (d) Ventana inválida (fin <= inicio) -> se procesa el clip completo.
    frames_completos, _ = main.extraer_fotogramas_base64(
        str(ruta_test), 5, ventana=(3.0, 3.0)
    )
    comprobar(
        len(frames_completos) == 5,
        f"Ventana inválida -> clip completo (obtenidos: {len(frames_completos)})",
    )

    # (e) Recorte estricto del FIN: el último fotograma extraído debe ser el
    # de frame_end = int(3.0 * fps) — jamás uno posterior al fin del rango.
    img_fin = _cv2.imdecode(
        _np.frombuffer(_b64.b64decode(frames_ventana[-1]), _np.uint8), _cv2.IMREAD_COLOR
    )
    cap = _cv2.VideoCapture(str(ruta_test))
    fps_test = cap.get(_cv2.CAP_PROP_FPS) or 25.0
    frame_end_esperado = int(3.0 * fps_test)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_end_esperado)
    leido_fin, ref_fin = cap.read()
    cap.release()
    if leido_fin and ref_fin is not None:
        ref_fin = _cv2.resize(
            ref_fin,
            (main.ANCHO_FOTOGRAMA, main.ALTO_FOTOGRAMA),
            interpolation=_cv2.INTER_AREA,
        )
        diff_fin = float(_np.mean(_np.abs(img_fin.astype("int16") - ref_fin.astype("int16"))))
        comprobar(
            diff_fin < 6.0,
            f"Recorte estricto: el último frame es frame_end={frame_end_esperado} (diff={diff_fin:.2f})",
        )
    else:
        comprobar(False, "No se pudo leer el frame de referencia del fin del rango")

# --- Fallback seguro: la IA nunca rompe la interfaz --------------------------
# La contingencia es por motor: 'nube' usa el mensaje que pide acortar el
# rango ("No determinado"); 'local' mantiene el genérico de reintento.
motor_original = main._consultar_motor


def motor_caido(*_args, **_kwargs):
    raise main.HTTPException(status_code=502, detail="Gemini saturado (simulado)")


main._consultar_motor = motor_caido
try:
    informe_caido = main.generar_informe_json(["QUJD"], "ctx", "tel", motor="nube")
finally:
    main._consultar_motor = motor_original

comprobar(
    informe_caido.get("culpable") == "No determinado",
    f"Motor nube caído -> 'No determinado' (recibido: {informe_caido.get('culpable')!r})",
)
comprobar(informe_caido.get("sancion_sugerida") == "N/A", "Motor caído -> sanción 'N/A'")
comprobar(
    "acortar el rango" in informe_caido.get("analisis_temporal", ""),
    "Motor nube caído -> análisis temporal pide acortar el rango",
)

main._consultar_motor = motor_caido
try:
    informe_caido_local = main.generar_informe_json(["QUJD"], "ctx", "tel", motor="local")
finally:
    main._consultar_motor = motor_original

comprobar(
    informe_caido_local.get("culpable") == "Error de procesamiento",
    f"Motor local caído -> 'Error de procesamiento' (recibido: {informe_caido_local.get('culpable')!r})",
)
comprobar(
    "no pudo procesar este clip" in informe_caido_local.get("analisis_temporal", ""),
    "Motor local caído -> análisis temporal con mensaje seguro",
)


def motor_json_roto(*_args, **_kwargs):
    return "esto no es json {{{"


main._consultar_motor = motor_json_roto
try:
    informe_roto = main.generar_informe_json(["QUJD"], "ctx", "tel", motor="local")
finally:
    main._consultar_motor = motor_original
comprobar(
    informe_roto.get("culpable") == "Error de procesamiento",
    "JSON roto -> informe seguro por defecto",
)


def motor_vacio(*_args, **_kwargs):
    return "{}"


main._consultar_motor = motor_vacio
try:
    informe_vacio = main.generar_informe_json(["QUJD"], "ctx", "tel", motor="local")
finally:
    main._consultar_motor = motor_original
comprobar(informe_vacio.get("sancion_sugerida") == "N/A", "JSON vacío -> informe seguro por defecto")

for clave in ("analisis_temporal", "culpable", "justificacion_reglamentaria", "sancion_sugerida", "analisis_tecnico"):
    comprobar(clave in informe_caido, f"El informe seguro incluye '{clave}' (el frontend no se rompe)")

# ------------------------------------------------------------------ RESULTADO
print("\n" + "=" * 72)
print(f"RESULTADO: {len(APROBADOS)} comprobaciones OK / {len(FALLOS)} fallos")
if FALLOS:
    print("FALLOS:")
    for fallo in FALLOS:
        print("  - " + fallo)
    sys.exit(1)
print("VALIDACION_COMPLETA_OK")