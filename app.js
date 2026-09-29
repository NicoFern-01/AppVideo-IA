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

// ---------------------------------------------------------------------------
// Constantes y estado
// ---------------------------------------------------------------------------
const LIMITE_MB = 15;
const LIMITE_BYTES = LIMITE_MB * 1024 * 1024;
const FORMATOS = ["mp4", "mov"];

let archivoActual = null;   // Archivo (File) seleccionado por el usuario
let urlVistaPrevia = null;  // ObjectURL activa del reproductor

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

  if (archivo.size > LIMITE_BYTES) {
    limpiarArchivo();
    mostrarAviso(
      `El clip pesa ${formatearPeso(archivo.size)} y el máximo es de ${LIMITE_MB} MB. ` +
      "Recorta un clip de 5 a 10 segundos con el momento exacto del toque.",
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
