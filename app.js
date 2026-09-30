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
const rangoInicio = document.getElementById("rango-inicio");
const rangoFin = document.getElementById("rango-fin");
const trimmerTiempos = document.getElementById("trimmer-tiempos");
const trimmerDisplayInicio = document.getElementById("trimmer-display-inicio");
const trimmerDisplayFin = document.getElementById("trimmer-display-fin");
const trimmerDisplayDuracion = document.getElementById("trimmer-display-duracion");

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
// Trimmer: dos barras de rango delimitan los segundos de la maniobra.
// REGLAS: (1) las etiquetas se actualizan SOLO en 'input' (tiempo real,
// aun con el video pausado); 'timeupdate' solo aplica el bucle acotado.
// (2) Inicio y Fin tienen manejadores SEPARADOS (sin conflicto).
// ---------------------------------------------------------------------------

// Separación mínima Inicio < Fin y tolerancia anti-loop de Chromium.
const SEPARACION_MINIMA = 0.5;   // El Inicio se detiene 0.5 s antes del Fin.
const TOLERANCIA_BUCLE = 0.1;    // Margen para no atrapar a Chromium en un loop.
let suprimirBucleHasta = 0;      // Supresor temporal tras previsualizar el Fin.
let temporizadorVistaFin = null; // Anti-rebote de la vista previa del Fin.

/** Formatea segundos como "02.4s" (parte entera con 2 dígitos). */
function formatearTiempo(segundos) {
  return `${Number(segundos || 0).toFixed(1).padStart(4, "0")}s`;
}

/** Formatea segundos como "02.4 seg" para la línea de tiempo gigante. */
function formatearTiempoLargo(segundos) {
  return `${Number(segundos || 0).toFixed(1).padStart(4, "0")} seg`;
}

/** Refresca la etiqueta dinámica gigante "Inicio: XX.X seg | Fin: XX.X seg". */
function actualizarTrimmer() {
  const inicio = Number(rangoInicio.value || 0);
  const fin = Number(rangoFin.value || 0);
  const duracion = Math.max(0, fin - inicio);
  trimmerTiempos.textContent =
    `Inicio: ${formatearTiempo(inicio)} | Fin: ${formatearTiempo(fin)}`;
  if (trimmerDisplayInicio) {
    trimmerDisplayInicio.textContent = `Inicio: ${formatearTiempoLargo(inicio)}`;
  }
  if (trimmerDisplayFin) {
    trimmerDisplayFin.textContent = `Fin: ${formatearTiempoLargo(fin)}`;
  }
  if (trimmerDisplayDuracion) {
    trimmerDisplayDuracion.textContent = `· Fragmento: ${formatearTiempoLargo(duracion)}`;
  }
}

/** Salto seguro: recorta al rango [0, duración] antes de asignar currentTime. */
function saltarA(segundos) {
  if (!Number.isFinite(Number(reproductor.duration))) return;
  const destino = Math.min(Math.max(Number(segundos) || 0, 0), reproductor.duration);
  try {
    reproductor.currentTime = destino;
  } catch {
    /* Metadata aún no lista en algún navegador: se ignora sin romper nada. */
  }
}

/** Cuando el navegador conoce la duración, calibra las barras al clip real. */
function prepararTrimmer() {
  const duracion = Number(reproductor.duration);
  if (!Number.isFinite(duracion) || duracion <= 0) return;
  for (const barra of [rangoInicio, rangoFin]) {
    barra.min = "0";
    barra.max = duracion.toFixed(1);
    barra.step = "0.1";
  }
  rangoInicio.value = "0";
  rangoFin.value = duracion.toFixed(1);
  actualizarTrimmer();
}

/** Estado neutro del Trimmer (al cargar o quitar un clip). */
function reiniciarTrimmer() {
  if (temporizadorVistaFin !== null) {
    clearTimeout(temporizadorVistaFin);
    temporizadorVistaFin = null;
  }
  suprimirBucleHasta = 0;
  rangoInicio.value = "0";
  rangoFin.value = "0";
  if (trimmerDisplayInicio) trimmerDisplayInicio.textContent = "Inicio: 00.0 seg";
  if (trimmerDisplayFin) trimmerDisplayFin.textContent = "Fin: 00.0 seg";
  if (trimmerDisplayDuracion) trimmerDisplayDuracion.textContent = "· Fragmento: 00.0 seg";
  trimmerTiempos.textContent = "Inicio: 00.0s | Fin: 00.0s";
}

/** Al mover INICIO: actualiza etiquetas al instante, valida límite y previsualiza.
 *  Pausa el video para mostrar el cuadro exacto de comienzo (saltar + mantener
 *  la pausa si estaba pausado), así se ve el frame fijo sin conflicto. */
function alMoverInicio() {
  let valor = Number(rangoInicio.value);
  if (!Number.isFinite(valor)) return;

  // Validación de límites: el Inicio nunca puede ser >= Fin: se detiene
  // 0.5 s antes del final (SEPARACION_MINIMA).
  const fin = Number(rangoFin.value);
  if (Number.isFinite(fin) && valor >= fin) {
    valor = Math.max(0, fin - SEPARACION_MINIMA);
    rangoInicio.value = String(valor);
  }

  actualizarTrimmer(); // Etiquetas en tiempo real, aun con el video pausado.

  // Previsualización: salta al segundo exacto de comienzo y pausa
  // temporalmente para que se vea el cuadro fijo.
  if (!reproductor.paused) reproductor.pause();
  saltarA(valor);
}

/** Al mover FIN: actualiza etiquetas al instante, valida límite y previsualiza.
 *  El salto al final se difiere con anti-rebote para no entrar en bucle con
 *  el validador de tiempo ('timeupdate'): mientras el usuario arrastra, solo
 *  se refrescan las etiquetas; al soltar (250 ms sin mover), se previsualiza
 *  el cuadro de fin con el bucle suprimido temporalmente. */
function alMoverFin() {
  let valor = Number(rangoFin.value);
  if (!Number.isFinite(valor)) return;

  // Validación de límites: el Fin nunca puede ser <= Inicio: se fuerza a
  // Inicio + 0.5 s como mínimo (SEPARACION_MINIMA).
  const inicio = Number(rangoInicio.value);
  if (Number.isFinite(inicio) && valor <= inicio) {
    valor = inicio + SEPARACION_MINIMA;
    rangoFin.value = String(valor);
  }

  actualizarTrimmer(); // Etiquetas en tiempo real, aun con el video pausado.

  // Anti-rebote: retrasa la vista previa 250 ms para que el arrastre no
  // dispare saltos continuos que choquen con 'timeupdate'.
  if (temporizadorVistaFin !== null) clearTimeout(temporizadorVistaFin);
  temporizadorVistaFin = setTimeout(() => {
    temporizadorVistaFin = null;
    if (!reproductor.paused) reproductor.pause();
    suprimirBucleHasta = performance.now() + 600; // El bucle ignora este salto.
    saltarA(valor);
  }, 250);
}

/** Compatibilidad: despacha al manejador separado según la barra movida. */
function alMoverRango(evento) {
  if (evento.currentTarget === rangoInicio) alMoverInicio();
  else alMoverFin();
}

/** Bucle acotado ESTRICTO con tolerancia anti-loop de Chromium (0.1 s):
 *  Si currentTime sale de la ventana [Inicio, Fin], vuelve al Inicio.
 *  La tolerancia evita el loop infinito al forzar 'currentTime' y el
 *  supresor ignora el salto de previsualización del Fin. */
function alActualizarTiempo() {
  if (reproductor.paused) return;
  if (performance.now() < suprimirBucleHasta) return; // Salto de vista previa.
  const fin = Number(rangoFin.value);
  const inicio = Number(rangoInicio.value);
  if (!Number.isFinite(fin) || !Number.isFinite(inicio)) return;
  const ahora = Number(reproductor.currentTime);
  if (!Number.isFinite(ahora)) return;
  if (ahora >= fin - TOLERANCIA_BUCLE || ahora < inicio - TOLERANCIA_BUCLE) {
    saltarA(inicio); // Reinicia igualando la posición al Tiempo Inicio.
  }
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
  reiniciarTrimmer(); // Las barras se calibran al cargar el metadata del nuevo clip

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
  reiniciarTrimmer();
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

// --- Trimmer: metadata del video + barras de rango + bucle acotado -------
reproductor.addEventListener("loadedmetadata", prepararTrimmer);
reproductor.addEventListener("timeupdate", alActualizarTiempo);
rangoInicio.addEventListener("input", alMoverRango);
rangoFin.addEventListener("input", alMoverRango);
reiniciarTrimmer(); // Estado inicial neutro

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

  // Recorte elegido en el Trimmer -> 'tiempo_inicio' / 'tiempo_fin' en segundos.
  // Si el usuario no tocó las barras (fin = 0), el backend usa el clip completo.
  formulario.append("tiempo_inicio", Number(rangoInicio.value || 0).toFixed(1));
  formulario.append("tiempo_fin", Number(rangoFin.value || 0).toFixed(1));

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
