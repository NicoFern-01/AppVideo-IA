/* ==========================================================================
   app.js — 🏁 Race Control
   Interactividad en JavaScript nativo con Fetch API:
   selección/drag & drop del clip, previsualización, gestión de estados de
   carga (spinner) y renderizado del informe JSON sin recargar la página.
   ========================================================================== */

"use strict";

// ---------------------------------------------------------------------------
// Referencias al DOM
// ---------------------------------------------------------------------------
const zonaCarga = document.getElementById("zona-carga");
const inputArchivo = document.getElementById("input-archivo");
const aviso = document.getElementById("aviso");
const previsualizacion = document.getElementById("previsualizacion");
const reproductor = document.getElementById("reproductor");
const datoNombre = document.getElementById("dato-nombre");
const datoPeso = document.getElementById("dato-peso");
const btnQuitar = document.getElementById("btn-quitar");
const btnAnalizar = document.getElementById("btn-analizar");
const panelSpinner = document.getElementById("panel-spinner");
const panelError = document.getElementById("panel-error");
const textoError = document.getElementById("texto-error");
const btnReintentar = document.getElementById("btn-reintentar");
const panelResultados = document.getElementById("panel-resultados");
const resultadoCulpable = document.getElementById("resultado-culpable");
const resultadoAnalisis = document.getElementById("resultado-analisis");
const resultadoSancion = document.getElementById("resultado-sancion");
const btnNuevo = document.getElementById("btn-nuevo");
const campoContexto = document.getElementById("contexto-incidente");
const versionApp = document.getElementById("version-app");
const radiosMotor = document.querySelectorAll('input[name="motor"]');
const consejoArchivo = document.getElementById("consejo-archivo");
const spinnerTexto = document.getElementById("spinner-texto");
const timelineRuler = document.getElementById("timeline-ruler");
const timelineLanes = document.getElementById("timeline-lanes");
const timelineTrack = document.getElementById("timeline-track");
const timelinePlay = document.getElementById("timeline-play");
const btnFullscreen = document.getElementById("btn-fullscreen");
const selectionOverlay = document.getElementById("selection-overlay");
const playheadLine = document.getElementById("playhead-line");
const tlInicio = document.getElementById("tl-inicio");
const tlFin = document.getElementById("tl-fin");
const tlFragmento = document.getElementById("tl-fragmento");
const tlPosicion = document.getElementById("tl-posicion");
const selEtiquetaInicio = document.getElementById("sel-etiqueta-inicio");
const selEtiquetaFin = document.getElementById("sel-etiqueta-fin");
const btnVer = document.getElementById("timeline-ver");
const btnBloquear = document.getElementById("timeline-bloquear");

// ---------------------------------------------------------------------------
// Constantes y estado
// ---------------------------------------------------------------------------
// Configuración por motor de análisis (espejo de las constantes de main.py):
// Local (Ollama) cuida la VRAM; Nube (Gemini) admite videos largos y más fotogramas.
const MOTORES = {
  local: {
    maxMb: 15,
    consejo:
      "Formatos <strong>MP4 / MOV</strong> · máximo <strong>15 MB</strong> · recorte recomendado de 5 a 10 segundos",
    spinner:
      "Analizando telemetría visual en tu equipo... (puede tardar de 30 a 90 segundos por hardware local)",
  },
  nube: {
    maxMb: 100,
    consejo:
      "Formatos <strong>MP4 / MOV</strong> · hasta <strong>100 MB</strong> · en modo Nube puedes subir " +
      "videos de <strong>20 a 45 segundos</strong> con total libertad",
    spinner:
      "Analizando incidente en la nube (Gemini)... (suele tardar entre 20 y 60 segundos)",
  },
};
const FORMATOS = ["mp4", "mov"];

let archivoActual = null;   // Archivo (File) seleccionado por el usuario
let urlVistaPrevia = null;  // ObjectURL activa del reproductor

/** Devuelve el motor marcado en la interfaz ("local" por defecto). */
function motorActual() {
  const marcado = document.querySelector('input[name="motor"]:checked');
  return marcado?.value === "nube" ? "nube" : "local";
}

/** Configuración (límites y textos) del motor activo. */
function configMotor() {
  return MOTORES[motorActual()];
}

// ---------------------------------------------------------------------------
// Timeline window: línea de tiempo profesional (estilo VSDC / Premiere) que
// delimita los segundos exactos de la maniobra sobre la pista de video.
// REGLAS: (1) el bloque verde actualiza 'tiempoInicio'/'tiempoFin' al ms
// exacto en cada arrastre; (2) al mover un borde se INYECTA el fotograma
// (currentTime + play()/pause() inmediato) para que el navegador repinte el
// cuadro aunque el video esté pausado; (3) el bucle estricto NO depende de
// 'timeupdate' (el navegador lo dispara cada ~250 ms y el video "escapa"):
// un motor unificado a 16 ms + rAF verifica cada instante que la posición
// siga dentro de [Inicio - 0.1, Fin) mientras reproduce, y si se escapa
// vuelve al Inicio obligatoriamente.
// ---------------------------------------------------------------------------

const SEPARACION_MINIMA = 0.5;    // El Inicio se detiene 0.5 s antes del Fin.
const INTERVALO_BUCLE_MS = 16;    // Motor de control del rango a ~60 FPS (16 ms/tic).
const TOLERANCIA_INFERIOR = 0.1;  // Bucle estricto: cuenta como fuga solo cuando
                                  // 'ahora < inicio - 0.1'; por debajo, la rejilla
                                  // de fotogramas (FPS raros) sigue siendo "dentro".
const TOLERANCIA_SEEK = 0.05;     // 50 ms de enfriamiento entre saltos forzados: evita
                                  // que el navegador quede atrapado saltando eterna-
                                  // mente al mismo píxel temporal (bucle de seeks).

let tiempoInicio = 0;   // Segundos del borde izquierdo del bloque verde
let tiempoFin = 0;      // Segundos del borde derecho del bloque verde
let seekPendiente = false;      // true mientras el motor procesa un salto forzado
let ultimoSaltoForzadoEn = 0;   // performance.now() del último salto forzado
let inyeccionActiva = false;    // true durante la micro-reproducción play()/pause()
let arrastreSeleccion = null; // Arrastre activo del bloque verde o de sus bordes
let pistaBloqueada = false;   // Candado de la capa "Video Track 1"

/** Convierte segundos a "05.123" (2 dígitos + milisegundos exactos). */
function formatearTiempo(segundos) {
  return Number(segundos || 0).toFixed(3).padStart(6, "0");
}

/** Convierte segundos a "00:05.123" (mm:ss.mmm) para la regla milimétrica. */
function formatearTiempoRegla(segundos) {
  const totalMs = Math.round(Math.max(0, Number(segundos) || 0) * 1000);
  const minutos = Math.floor(totalMs / 60000);
  const segundosEnteros = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return (
    `${String(minutos).padStart(2, "0")}:` +
    `${String(segundosEnteros).padStart(2, "0")}.` +
    String(ms).padStart(3, "0")
  );
}

/** Duración total del clip cargado (0 si aún no hay metadata). */
function duracionTotal() {
  const duracion = Number(reproductor.duration);
  return Number.isFinite(duracion) && duracion > 0 ? duracion : 0;
}

/** Redondea a milisegundos exactos (la precisión que viaja al backend). */
function redondearMs(valor) {
  return Math.round((Number(valor) || 0) * 1000) / 1000;
}

/** Refresca la lectura de la barra de herramientas y coloca el bloque verde
 *  sobre la pista con precisión de milisegundos: posición% = (t / duración) * 100. */
function actualizarTimeline() {
  const duracion = duracionTotal();
  const hayClip = duracion > 0;

  if (selectionOverlay) {
    if (hayClip) {
      const pInicio = (Math.min(Math.max(tiempoInicio, 0), duracion) / duracion) * 100;
      const pFin = (Math.min(Math.max(tiempoFin, 0), duracion) / duracion) * 100;
      selectionOverlay.style.left = `${Math.min(pInicio, pFin)}%`;
      selectionOverlay.style.width = `${Math.max(0, pFin - pInicio)}%`;
      selectionOverlay.hidden = false;
    } else {
      selectionOverlay.hidden = true;
    }
  }
  if (tlInicio) tlInicio.textContent = `${formatearTiempo(tiempoInicio)} s`;
  if (tlFin) tlFin.textContent = `${formatearTiempo(tiempoFin)} s`;
  if (tlFragmento) {
    tlFragmento.textContent = `${formatearTiempo(Math.max(0, tiempoFin - tiempoInicio))} s`;
  }
  if (selEtiquetaInicio) selEtiquetaInicio.textContent = formatearTiempo(tiempoInicio);
  if (selEtiquetaFin) selEtiquetaFin.textContent = formatearTiempo(tiempoFin);
}

/** Mueve la aguja roja con (video.currentTime / video.duration) * 100 y
 *  refresca la lectura POS de la barra de herramientas. */
function actualizarAguja() {
  if (!playheadLine) return;
  const duracion = duracionTotal();
  if (duracion <= 0) {
    playheadLine.hidden = true;
    if (tlPosicion) tlPosicion.textContent = `${formatearTiempo(0)} s`;
    return;
  }
  const ahora = Math.min(Math.max(Number(reproductor.currentTime) || 0, 0), duracion);
  playheadLine.hidden = false;
  playheadLine.style.left = `${(ahora / duracion) * 100}%`;
  if (tlPosicion) tlPosicion.textContent = `${formatearTiempo(ahora)} s`;
}

/** La aguja roja se recalcula en CADA frame para que el movimiento sea fluido. */
function bucleAguja() {
  actualizarAguja();
  requestAnimationFrame(bucleAguja);
}

/** Construye la regla superior: marcas uniformes etiquetadas (mm:ss.mmm) y
 *  marcas menores a medio paso, repartidas según la duración total del clip. */
function construirRegla() {
  if (!timelineRuler) return;
  timelineRuler.textContent = "";
  const duracion = duracionTotal();
  if (duracion <= 0) return;

  const pasos = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const paso = pasos.find((p) => duracion / p <= 10) || pasos[pasos.length - 1];

  const crearMarca = (tiempo, esMenor) => {
    const marca = document.createElement("span");
    marca.className =
      "timeline-ruler-marca" + (esMenor ? " timeline-ruler-marca--menor" : "");
    if (tiempo <= 0) marca.classList.add("timeline-ruler-marca--inicio");
    if (tiempo >= duracion) marca.classList.add("timeline-ruler-marca--fin");
    marca.style.left = `${Math.min(Math.max(tiempo / duracion, 0), 1) * 100}%`;
    if (!esMenor) {
      const etiqueta = document.createElement("span");
      etiqueta.className = "timeline-ruler-etiqueta";
      etiqueta.textContent = formatearTiempoRegla(tiempo);
      marca.appendChild(etiqueta);
    }
    timelineRuler.appendChild(marca);
  };

  const tramos = Math.floor(duracion / paso);
  for (let i = 0; i <= tramos; i++) {
    const t = i * paso;
    crearMarca(t, false);
    if (t + paso / 2 < duracion) crearMarca(t + paso / 2, true);
  }
  if (duracion - tramos * paso > paso * 0.4) crearMarca(duracion, false); // Cierre exacto.
}

/** Salto seguro: recorta al rango [0, duración] antes de asignar currentTime. */
function saltarA(segundos) {
  if (!Number.isFinite(parseFloat(reproductor.duration))) return;
  const destino = Math.min(Math.max(parseFloat(segundos) || 0, 0), reproductor.duration);
  try {
    reproductor.currentTime = destino;
  } catch {
    /* Metadata aún no lista en algún navegador: se ignora sin romper nada. */
  }
}

/** Cuando el navegador conoce la duración, calibra la línea de tiempo al clip
 *  real: bloque verde al 100 %, regla milimétrica y aguja al inicio. */
function prepararTimeline() {
  const duracion = duracionTotal();
  if (duracion <= 0) return;
  tiempoInicio = 0;
  tiempoFin = redondearMs(duracion);
  rearmarBucle();
  construirRegla();
  actualizarTimeline();
  actualizarAguja();
}

/** Estado neutro de la línea de tiempo (al cargar o quitar un clip). */
function reiniciarTimeline() {
  tiempoInicio = 0;
  tiempoFin = 0;
  rearmarBucle();
  arrastreSeleccion = null;
  if (timelineRuler) timelineRuler.textContent = "";
  actualizarTimeline();
  actualizarAguja();
  actualizarBotonPlay();
}

/** Convierte una posición horizontal (px) de la línea de tiempo a segundos. */
function tiempoDesdeX(clientX) {
  const duracion = duracionTotal();
  if (!timelineLanes || duracion <= 0) return 0;
  const rect = timelineLanes.getBoundingClientRect();
  if (rect.width <= 0) return 0;
  const fraccion = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
  return fraccion * duracion;
}

/** Pointerdown sobre el bloque verde o sus bordes: inicia el arrastre y, como
 *  en los editores reales, pausa el video para trabajar sobre un fotograma. */
function alIniciarArrastre(evento) {
  if (!archivoActual || pistaBloqueada) return;
  if (duracionTotal() <= 0) return;
  evento.preventDefault();
  const destino = evento.target;
  const modo =
    destino && destino.dataset && destino.dataset.modo ? destino.dataset.modo : "mover";
  arrastreSeleccion = {
    id: evento.pointerId,
    modo,
    desfase: modo === "mover" ? tiempoDesdeX(evento.clientX) - tiempoInicio : 0,
    ancho: Math.max(0, tiempoFin - tiempoInicio),
  };
  try {
    selectionOverlay.setPointerCapture(evento.pointerId);
  } catch {
    /* Captura no disponible: el arrastre sigue con los eventos normales. */
  }
  if (!reproductor.paused) reproductor.pause();
  // Previsualización inmediata del borde que se va a tocar (inyección de fotograma).
  inyectarFotograma(modo === "fin" ? tiempoFin : tiempoInicio);
}

/** Pointermove: actualiza 'tiempoInicio'/'tiempoFin' al milisegundo exacto
 *  (toFixed(3)) y salta el reproductor superior al fotograma resultante. */
function alArrastrarSeleccion(evento) {
  if (!arrastreSeleccion || evento.pointerId !== arrastreSeleccion.id) return;
  evento.preventDefault();
  const duracion = duracionTotal();
  if (duracion <= 0) return;
  const t = tiempoDesdeX(evento.clientX);

  if (arrastreSeleccion.modo === "mover") {
    const inicio = Math.min(
      Math.max(t - arrastreSeleccion.desfase, 0),
      Math.max(duracion - arrastreSeleccion.ancho, 0)
    );
    tiempoInicio = redondearMs(inicio);
    tiempoFin = redondearMs(inicio + arrastreSeleccion.ancho);
  } else if (arrastreSeleccion.modo === "inicio") {
    // El Inicio jamás alcanza al Fin (SEPARACION_MINIMA).
    tiempoInicio = redondearMs(Math.max(0, Math.min(t, tiempoFin - SEPARACION_MINIMA)));
  } else {
    // El Fin jamás se queda atrás del Inicio.
    tiempoFin = redondearMs(Math.min(duracion, Math.max(t, tiempoInicio + SEPARACION_MINIMA)));
  }
  // Doble candado: ambos bordes siempre dentro de [0, duración] al ms exacto.
  tiempoInicio = redondearMs(Math.min(Math.max(tiempoInicio, 0), duracion));
  tiempoFin = redondearMs(Math.min(Math.max(tiempoFin, 0), duracion));

  actualizarTimeline();

  // FOTOGRAMA EN TIEMPO REAL: la pantalla superior refleja matemáticamente
  // dónde empieza (o termina) el corte, con inyección forzada del cuadro.
  inyectarFotograma(arrastreSeleccion.modo === "fin" ? tiempoFin : tiempoInicio);
  actualizarAguja();
}

/** Pointerup/cancel: cierra el arrastre; el rango ya quedó al milisegundo. */
function alSoltarSeleccion(evento) {
  if (!arrastreSeleccion || evento.pointerId !== arrastreSeleccion.id) return;
  try {
    selectionOverlay.releasePointerCapture(evento.pointerId);
  } catch {
    /* La captura ya se había liberado: nada que hacer. */
  }
  arrastreSeleccion = null;
}

/** Clic en la regla: mueve la aguja roja y salta el video a ese tiempo. */
function alClicRegla(evento) {
  const duracion = duracionTotal();
  if (duracion <= 0 || !timelineRuler) return;
  const rect = timelineRuler.getBoundingClientRect();
  if (rect.width <= 0) return;
  const fraccion = Math.min(Math.max((evento.clientX - rect.left) / rect.width, 0), 1);
  // Como en un editor NLE: fijar el cursor en la regla pausa la reproducción
  // e inyecta el fotograma elegido para verlo al instante.
  if (!reproductor.paused) reproductor.pause();
  inyectarFotograma(fraccion * duracion);
  actualizarAguja();
}

/** Alterna Play/Pausa del tramo: al reproducir arranca en el Inicio si la
 *  posición actual quedó fuera de la ventana [Inicio, Fin]. */
function alternarPlayTimeline() {
  if (!archivoActual) return;
  if (reproductor.paused) {
    // parseFloat estricto antes de CUALQUIER comparación matemática.
    const inicio = parseFloat(tiempoInicio);
    const fin = parseFloat(tiempoFin);
    const ahora = parseFloat(reproductor.currentTime);
    rearmarBucle();
    if (!(Number.isFinite(inicio) && ahora >= inicio && ahora < fin)) saltarA(inicio);
    const intento = reproductor.play();
    if (intento && typeof intento.catch === "function") {
      intento.catch(() => { /* Autoplay bloqueado u otro fallo: se ignora. */ });
    }
  } else {
    reproductor.pause();
  }
}

/** Sincroniza el icono del botón Play/Pausa de la barra de herramientas. */
function actualizarBotonPlay() {
  if (!timelinePlay || inyeccionActiva) return; // Ignora el play()/pause() de la inyección
  const reproduciendo = !reproductor.paused && !reproductor.ended;
  timelinePlay.textContent = reproduciendo ? "⏸" : "▶";
  timelinePlay.classList.toggle("timeline-play--activo", reproduciendo);
  timelinePlay.setAttribute(
    "aria-label",
    reproduciendo ? "Pausar la reproducción" : "Reproducir el tramo seleccionado"
  );
}

/** El clip completo alcanzó su fin: el bucle acotado lo devuelve al Inicio. */
function alTerminarBucle() {
  const inicio = parseFloat(tiempoInicio);
  const fin = parseFloat(tiempoFin);
  if (!archivoActual || !(fin > inicio)) return;
  rearmarBucle();
  saltarA(inicio);
  const intento = reproductor.play();
  if (intento && typeof intento.catch === "function") {
    intento.catch(() => { /* Reproducción automática bloqueada: se ignora. */ });
  }
}

// ---------------------------------------------------------------------------
// MOTOR DE REPRODUCCIÓN UNIFICADO (alta frecuencia, sin 'timeupdate')
// ---------------------------------------------------------------------------
// El evento 'timeupdate' queda DESCARTADO como control de bucle: el navegador
// lo dispara solo cada ~250 ms y entre evento y evento el video escapa del
// rango. En su lugar, 'tickMotor' se ejecuta cada 16 ms (setInterval, ~60 FPS)
// —también con la pestaña en segundo plano, donde rAF se suspende— y en cada
// requestAnimationFrame, aplicando la regla estricta:
//     if (video.currentTime >= tiempoFin || video.currentTime < (tiempoInicio - 0.1))
//         video.currentTime = tiempoInicio;

/** Rearma el bucle: borra el seek pendiente y el enfriamiento anti-bucle. */
function rearmarBucle() {
  seekPendiente = false;
  ultimoSaltoForzadoEn = 0;
}

/** Salto obligatorio al Inicio con marca de "seek en curso" ('seekPendiente')
 *  y red de seguridad: si el navegador nunca emite 'seeked', el bucle se rearma
 *  solo a los 300 ms para no quedarse muerto. */
function forzarSaltoInicio(inicio) {
  seekPendiente = true;
  ultimoSaltoForzadoEn = performance.now();
  try {
    reproductor.currentTime = inicio;
  } catch {
    /* Metadata aún no lista en algún navegador: se ignora sin romper nada. */
  }
  setTimeout(() => {
    seekPendiente = false;
  }, 300); // Watchdog: sin 'seeked' el motor no puede quedarse bloqueado.
}

/** TRUCO DE INYECCIÓN DE FOTOGRAMA (seeking fix): fija el tiempo exacto y, si
 *  el video está pausado, obliga al motor del navegador a repintar el cuadro
 *  con una micro-reproducción play() → pause() inmediato. La bandera
 *  'inyeccionActiva' evita que esos play/pause muevan la interfaz ni el bucle. */
function inyectarFotograma(segundos) {
  const duracion = duracionTotal();
  if (duracion <= 0) return;
  const destino = Math.min(Math.max(parseFloat(segundos) || 0, 0), duracion);
  rearmarBucle();
  try {
    reproductor.currentTime = destino;
  } catch {
    /* Metadata aún no lista en algún navegador: se ignora sin romper nada. */
  }
  if (!reproductor.paused) return; // Si reproduce, el propio motor ya pinta.

  inyeccionActiva = true;
  const intento = reproductor.play();
  if (intento && typeof intento.catch === "function") {
    intento.catch(() => { /* Abortado por el pause() inmediato: es el diseño. */ });
  }
  reproductor.pause();
  // Red por si el navegador no emite 'pause': a los 50 ms se libera la bandera.
  setTimeout(() => {
    if (inyeccionActiva) {
      inyeccionActiva = false;
      actualizarBotonPlay();
    }
  }, 50);
}

/** Núcleo del bucle estricto de 16 ms. Comparaciones SIEMPRE con parseFloat
 *  estricto y doble tolerancia (inferior 0.1 s por rejilla de FPS + enfriamiento
 *  de 0.05 s entre saltos) para que el navegador jamás quede atrapado en un
 *  bucle infinito de seeks sobre el mismo instante temporal. */
function aplicarBucleEstricto() {
  if (reproductor.paused || reproductor.ended) return; // Solo manda al REPRODUCIR.
  if (inyeccionActiva || seekPendiente || reproductor.seeking) return;
  const inicio = parseFloat(tiempoInicio);
  const fin = parseFloat(tiempoFin);
  if (!Number.isFinite(inicio) || !Number.isFinite(fin) || !(fin > inicio)) return;
  const ahora = parseFloat(reproductor.currentTime);
  if (!Number.isFinite(ahora)) return;

  if (ahora >= fin || ahora < inicio - TOLERANCIA_INFERIOR) {
    if (performance.now() - ultimoSaltoForzadoEn < TOLERANCIA_SEEK * 1000) return;
    forzarSaltoInicio(inicio);
  }
}

/** Un tic del motor: aguja al día + control estricto del rango (60 FPS). */
function tickMotor() {
  actualizarAguja();
  aplicarBucleEstricto();
}

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------

/** Formatea unos bytes como "X.X MB". */
function formatearPeso(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Devuelve true si la extensión del nombre es .mp4 o .mov. */
function esFormatoValido(nombre) {
  const partes = nombre.toLowerCase().split(".");
  return FORMATOS.includes(partes[partes.length - 1]);
}

/** Muestra u oculta el aviso de la zona de carga (tipos: error | ok). */
function mostrarAviso(mensaje, tipo) {
  if (!mensaje) {
    aviso.hidden = true;
    aviso.textContent = "";
    return;
  }
  aviso.textContent = mensaje;
  aviso.className = `aviso aviso--${tipo}`;
  aviso.hidden = false;
}

/** Asigna la clase de color automático según la sanción recibida. */
function claseParaSancion(sancion) {
  const valor = (sancion || "").trim().toLowerCase();
  if (valor.includes("drive")) return "sancion sancion--drive-through"; // Rojo
  if (valor.includes("10")) return "sancion sancion--10s";              // Naranja
  if (valor.includes("5")) return "sancion sancion--5s";                // Amarillo
  if (valor.includes("incidente")) return "sancion sancion--incidente"; // Verde
  if (valor.includes("advertencia")) return "sancion sancion--advertencia"; // Azul
  return "sancion sancion--desconocida";                                // Gris
}

// ---------------------------------------------------------------------------
// Selección del archivo (selector nativo + arrastrar y soltar)
// ---------------------------------------------------------------------------

/** Valida formato/peso y carga el clip en el reproductor nativo. */
function cargarArchivo(archivo) {
  if (!archivo) return;

  if (!esFormatoValido(archivo.name)) {
    limpiarArchivo();
    mostrarAviso("Formato no admitido. Usa un clip con extensión .mp4 o .mov.", "error");
    return;
  }

  const { maxMb } = configMotor();
  if (archivo.size > maxMb * 1024 * 1024) {
    limpiarArchivo();
    const pista =
      motorActual() === "local"
        ? "Recorta un clip de 5 a 10 segundos o cambia al motor ☁️ Nube (Gemini), que admite videos de 20 a 45 s."
        : "Prueba con un clip más liviano o reduce la resolución del video.";
    mostrarAviso(
      `El clip pesa ${formatearPeso(archivo.size)} y el máximo del motor elegido es de ${maxMb} MB. ${pista}`,
      "error"
    );
    return;
  }

  archivoActual = archivo;
  if (urlVistaPrevia) URL.revokeObjectURL(urlVistaPrevia);
  urlVistaPrevia = URL.createObjectURL(archivo);

  reproductor.src = urlVistaPrevia;
  datoNombre.textContent = archivo.name;
  datoNombre.title = archivo.name;
  datoPeso.textContent = formatearPeso(archivo.size);
  previsualizacion.hidden = false;
  reiniciarTimeline(); // La regla se calibra al cargar el metadata del nuevo clip

  mostrarAviso("Clip listo ✅ Pulsa ANALIZAR INCIDENTE para iniciar el estudio.", "ok");
  btnAnalizar.disabled = false;
  panelResultados.hidden = true;
}

/** Limpia el archivo activo y devuelve la interfaz a su estado inicial. */
function limpiarArchivo() {
  archivoActual = null;
  if (urlVistaPrevia) {
    URL.revokeObjectURL(urlVistaPrevia);
    urlVistaPrevia = null;
  }
  reproductor.removeAttribute("src");
  reproductor.load();
  previsualizacion.hidden = true;
  btnAnalizar.disabled = true;
  reiniciarTimeline();
}

// Clic en la zona -> abre el selector nativo (el <label> ya lo gestiona solo)
zonaCarga.addEventListener("click", (evento) => {
  if (evento.target.tagName !== "LABEL") inputArchivo.click();
});
zonaCarga.addEventListener("keydown", (evento) => {
  if (evento.key === "Enter" || evento.key === " ") {
    evento.preventDefault();
    inputArchivo.click();
  }
});

inputArchivo.addEventListener("change", () => {
  cargarArchivo(inputArchivo.files[0]);
  inputArchivo.value = ""; // permite volver a elegir el mismo archivo
});

// --- Timeline: motor unificado (16 ms + rAF), inyección de fotograma y Play -
reproductor.addEventListener("loadedmetadata", prepararTimeline);
reproductor.addEventListener("play", () => {
  if (inyeccionActiva) return; // play() de la inyección: no tocar la interfaz
  actualizarBotonPlay();
});
reproductor.addEventListener("pause", () => {
  if (inyeccionActiva) inyeccionActiva = false; // Fin de la inyección forzada
  actualizarBotonPlay();
});
// Escucha 'seeking'/'seeked': garantiza que el buffer se pinte en pantalla y
// que el bucle estricto se rearma al terminar cada salto forzado.
reproductor.addEventListener("seeking", actualizarAguja);
reproductor.addEventListener("seeked", () => {
  seekPendiente = false;
  actualizarAguja();
});
reproductor.addEventListener("ended", alTerminarBucle);
reproductor.addEventListener("click", alternarPlayTimeline); // Sin controles nativos
if (timelinePlay) timelinePlay.addEventListener("click", alternarPlayTimeline);
if (timelineRuler) timelineRuler.addEventListener("click", alClicRegla);
if (selectionOverlay) {
  selectionOverlay.addEventListener("pointerdown", alIniciarArrastre);
  selectionOverlay.addEventListener("pointermove", alArrastrarSeleccion);
  selectionOverlay.addEventListener("pointerup", alSoltarSeleccion);
  selectionOverlay.addEventListener("pointercancel", alSoltarSeleccion);
}

// Panel de capa: ojo (atenuar la pista) y candado (bloquear el recorte).
if (btnVer && timelineTrack) {
  btnVer.addEventListener("click", () => {
    const oculta = timelineTrack.classList.toggle("timeline-track--oculta");
    btnVer.classList.toggle("timeline-layer-btn--activo", !oculta);
    btnVer.setAttribute("aria-pressed", String(!oculta));
  });
}
if (btnBloquear && timelineTrack) {
  btnBloquear.addEventListener("click", () => {
    pistaBloqueada = !pistaBloqueada;
    timelineTrack.classList.toggle("timeline-track--bloqueada", pistaBloqueada);
    btnBloquear.classList.toggle("timeline-layer-btn--activo", pistaBloqueada);
    btnBloquear.setAttribute("aria-pressed", String(pistaBloqueada));
  });
}

reiniciarTimeline(); // Estado inicial neutro
// Arranque del motor unificado: control estricto cada 16 ms (operativo incluso
// con la pestaña en segundo plano, donde rAF se suspende) + aguja roja fluida
// a la frecuencia de refresco de la pantalla.
setInterval(tickMotor, INTERVALO_BUCLE_MS);
requestAnimationFrame(bucleAguja);

// Eventos de arrastrar y soltar sobre la zona de carga
["dragenter", "dragover"].forEach((nombre) =>
  zonaCarga.addEventListener(nombre, (evento) => {
    evento.preventDefault();
    zonaCarga.classList.add("arrastrando");
  })
);
["dragleave", "drop"].forEach((nombre) =>
  zonaCarga.addEventListener(nombre, (evento) => {
    evento.preventDefault();
    zonaCarga.classList.remove("arrastrando");
  })
);
zonaCarga.addEventListener("drop", (evento) => {
  cargarArchivo(evento.dataTransfer.files[0]);
});

btnQuitar.addEventListener("click", () => {
  limpiarArchivo();
  mostrarAviso("", null);
});

// ---------------------------------------------------------------------------
// Selector de motor: refresca advertencias, límites y texto del spinner
// ---------------------------------------------------------------------------

/** Aplica los textos y límites del motor elegido (y revalida el clip cargado). */
function actualizarMotorUI() {
  const config = configMotor();
  consejoArchivo.innerHTML = config.consejo;
  spinnerTexto.textContent = config.spinner;

  if (archivoActual && archivoActual.size > config.maxMb * 1024 * 1024) {
    limpiarArchivo();
    mostrarAviso(
      `El clip cargado supera el máximo de ${config.maxMb} MB del motor elegido. ` +
      "Ajusta el motor o recorta el video.",
      "error"
    );
  }
}

radiosMotor.forEach((radio) => radio.addEventListener("change", actualizarMotorUI));
actualizarMotorUI(); // Estado inicial coherente con "Local" marcado

// ---------------------------------------------------------------------------
// Estados de carga (spinner / error / resultados)
// ---------------------------------------------------------------------------

/** Activa o desactiva el estado "procesando" de la interfaz. */
function estadoProcesando(activo) {
  panelSpinner.hidden = !activo;
  panelError.hidden = true;
  if (activo) panelResultados.hidden = true;
  btnAnalizar.disabled = activo || !archivoActual;
  btnAnalizar.classList.toggle("procesando", activo);
  document.body.setAttribute("aria-busy", String(activo));
}

/** Muestra el error devuelto por el backend en su panel. */
function mostrarError(mensaje) {
  textoError.textContent = mensaje;
  panelError.hidden = false;
  panelError.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---------------------------------------------------------------------------
// Envío del clip al backend (Fetch API) y render del informe
// ---------------------------------------------------------------------------

async function analizarIncidente() {
  if (!archivoActual) {
    mostrarAviso("Primero carga un clip MP4/MOV.", "error");
    return;
  }

  estadoProcesando(true);

  const formulario = new FormData();
  // La clave "video" coincide con el parámetro `video: UploadFile` de FastAPI
  formulario.append("video", archivoActual);

  // Contexto opcional ("Contexto del Incidente") -> parámetro `contexto` (Form)
  const textoContexto = campoContexto.value.trim();
  if (textoContexto) formulario.append("contexto", textoContexto);

  // Motor elegido en la interfaz -> parámetro `motor` (Form): "local" | "nube"
  formulario.append("motor", motorActual());

  // Recorte elegido en la Timeline -> 'tiempo_inicio' / 'tiempo_fin' en
  // segundos con precisión de milisegundos (toFixed(3)). Sin clip cargado
  // llegan 0.000 y el backend analiza el archivo completo.
  formulario.append("tiempo_inicio", (parseFloat(tiempoInicio) || 0).toFixed(3));
  formulario.append("tiempo_fin", (parseFloat(tiempoFin) || 0).toFixed(3));

  try {
    const respuesta = await fetch("/analizar", {
      method: "POST",
      body: formulario,
    });
    const datos = await respuesta.json().catch(() => null);

    if (!respuesta.ok) {
      throw new Error(datos?.detail || `El servidor respondió con código ${respuesta.status}.`);
    }

    renderizarInforme(datos);
  } catch (error) {
    mostrarError(
      error.message || "No se pudo contactar con el backend. ¿Está 'python main.py' en marcha?"
    );
  } finally {
    estadoProcesando(false);
  }
}

/** Pinta el JSON recibido en las tarjetas de resultados (sin recargar). */
function renderizarInforme(informe) {
  resultadoCulpable.textContent = informe.culpable || "No determinado";
  resultadoAnalisis.textContent = informe.analisis_tecnico || "No disponible.";
  resultadoSancion.textContent = informe.sancion_sugerida || "No determinada";
  resultadoSancion.className = claseParaSancion(informe.sancion_sugerida);

  panelResultados.hidden = false;
  panelResultados.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Pantalla Completa del video (Full Screen API nativa con variantes)
// ---------------------------------------------------------------------------

/** Elemento actualmente a pantalla completa (estándar + prefijos) o null. */
function elementoEnPantallaCompleta() {
  return (
    document.fullscreenElement ||
    document.webkitFullscreenElement ||
    document.mozFullScreenElement ||
    document.msFullscreenElement ||
    null
  );
}

/**
 * Alterna la Pantalla Completa sobre el <video> (pantalla limpia, sin la
 * interfaz encima). El motor del bucle estricto NO se detiene en fullscreen:
 * el `setInterval(tickMotor, INTERVALO_BUCLE_MS)` sigue latiendo en segundo
 * plano (el intervalo del motor jamás se limpia) y fuerza el regreso a
 * 'tiempoInicio' en cuanto el video supera 'tiempoFin'; rAF solo mueve la
 * aguja roja, por lo que la suspensión de rAF jamás afecta al bucle.
 */
function alternarPantallaCompleta() {
  try {
    if (elementoEnPantallaCompleta()) {
      const salida =
        document.exitFullscreen ||
        document.webkitExitFullscreen ||
        document.mozCancelFullScreen ||
        document.msExitFullscreen;
      if (salida) salida.call(document);
      return;
    }

    const solicitud =
      reproductor.requestFullscreen ||
      reproductor.webkitRequestFullscreen ||
      reproductor.webkitEnterFullscreen || // iOS Safari (solo sobre el vídeo)
      reproductor.mozRequestFullScreen ||
      reproductor.msRequestFullscreen;

    if (!solicitud) {
      mostrarAviso("Tu navegador no admite Pantalla Completa.", "error");
      return;
    }

    const promesa = solicitud.call(reproductor);
    if (promesa && typeof promesa.catch === "function") {
      promesa.catch(() =>
        mostrarAviso("El navegador denegó la Pantalla Completa.", "error")
      );
    }
  } catch (_error) {
    mostrarAviso("No se pudo activar la Pantalla Completa.", "error");
  }
}

btnFullscreen.addEventListener("click", alternarPantallaCompleta);
// Refuerzo del control estricto al entrar/salir de pantalla completa: tico
// inmediato del motor (el setInterval de 16 ms sigue activo igualmente).
document.addEventListener("fullscreenchange", () => tickMotor());
document.addEventListener("webkitfullscreenchange", () => tickMotor());

// ---------------------------------------------------------------------------
// Listeners de acción
// ---------------------------------------------------------------------------
btnAnalizar.addEventListener("click", analizarIncidente);

btnReintentar.addEventListener("click", () => {
  panelError.hidden = true;
  analizarIncidente();
});

btnNuevo.addEventListener("click", () => {
  limpiarArchivo();
  mostrarAviso("", null);
  panelError.hidden = true;
  panelResultados.hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ---------------------------------------------------------------------------
// Sello de versión del desarrollo (pie de página; fuente única: main.py)
// ---------------------------------------------------------------------------

/** Consulta GET /version y pinta "vX.Y.Z · fecha" en el pie de página. */
async function cargarVersion() {
  try {
    const respuesta = await fetch("/version", { cache: "no-store" });
    if (!respuesta.ok) throw new Error(String(respuesta.status));
    const datos = await respuesta.json();

    versionApp.textContent = `v${datos.version} · ${datos.fecha}`;
    const detalles = [
      datos.modelo_vlm ? `VLM: ${datos.modelo_vlm}` : null,
      datos.modelo_yolo ? `YOLO: ${datos.modelo_yolo}` : null,
      datos.rag ? "RAG: reglamento.txt" : "RAG: sin reglamento",
    ].filter(Boolean);
    versionApp.title = `Race Control v${datos.version} (${datos.fecha}) — ${detalles.join(" · ")}`;
  } catch {
    // Sin backend operativo el sello lo indica en vez de quedarse mudo.
    versionApp.textContent = "v— (backend sin conexión)";
  }
}

cargarVersion();
