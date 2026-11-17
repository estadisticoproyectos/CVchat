import os
import io
import datetime
import streamlit as st
import fitz  # PyMuPDF para leer PDFs
from docx import Document
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Analizador y Optimizador de CV con IA", page_icon="📄", layout="wide"
)

# Leer credenciales y modelo exclusivamente desde los Secrets de Streamlit
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
modelo_gemini = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")

# Blindaje contra errores de tipo lista en TOML/Secrets
if isinstance(modelo_gemini, list):
    modelo_gemini = modelo_gemini[0]
elif not modelo_gemini:
    modelo_gemini = "gemini-3.8-flash"

if not api_key:
    st.error("⚠️ Falta configurar `GEMINI_API_KEY` en los Secrets de Streamlit.")
    st.stop()

client = genai.Client(api_key=api_key)

# Inicializar estado de sesión (Session State) para renderizado progresivo y persistencia
defaults = {
    "etapa_completada": 0,  # 0: inicial, 1: afinidad lista, 2: diagnostico listo, 3: cv listo
    "texto_cv": "",
    "descripcion_trabajo": "",
    "porcentaje_afinidad": 0.0,
    "feedback_ia": "",
    "cv_mejorado": "",
    "descarga_realizada": False,
    "timestamp_descarga": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def extraer_texto_pdf(uploaded_file):
    """Extrae texto de un archivo PDF subido usando PyMuPDF."""
    texto = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def calcular_similitud_tfidf(texto_cv, texto_oferta):
    """Calcula la similitud de Coseno básica entre el CV y la oferta laboral."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([texto_cv, texto_oferta])
    similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similitud * 100, 2)


def llamar_gemini_con_respaldo(prompt, modelo_principal):
    """Ejecuta una llamada individual a Gemini con modelo de respaldo automático."""
    modelos_a_probar = list(
        dict.fromkeys([str(modelo_principal).strip(), "gemini-3.5-flash-lite"])
    )
    ultimo_error = None
    for modelo_actual in modelos_a_probar:
        try:
            resp = client.models.generate_content(
                model=modelo_actual, contents=prompt
            )
            if resp and resp.text:
                return resp.text
        except Exception as e:
            ultimo_error = e
            continue
    raise Exception(f"Todos los modelos disponibles fallaron. Último error: {ultimo_error}")


def generar_diagnostico_ia(texto_cv, texto_oferta, modelo):
    """Paso 2: Genera únicamente las recomendaciones y observaciones del reclutador."""
    prompt_diagnostico = f"""
    Actúa como un Reclutador Técnico experto y Especialista en Recursos Humanos.
    Analiza el siguiente Currículum Vitae en relación con la Descripción del Trabajo proporcionada.

    ---
    DESCRIPCIÓN DEL TRABAJO:
    {texto_oferta}
    ---
    CURRÍCULUM VITAE:
    {texto_cv}
    ---

    Entrega un análisis estructurado con:
    1. **Afinidad General y Diagnóstico:** Resumen de qué tan bien encaja el perfil.
    2. **Palabras Clave Faltantes (Hard/Soft Skills):** Tecnologías, herramientas o conceptos clave de la oferta ausentes en el CV.
    3. **Mejoras de Redacción y Verbos de Acción:** Cómo transformar viñetas pasivas en logros orientados a resultados.
    4. **Ideas Clave a Añadir:** Proyectos, certificaciones o enfoque a destacar.
    """
    return llamar_gemini_con_respaldo(prompt_diagnostico, modelo)


def generar_cv_reforzado_ia(texto_cv, texto_oferta, modelo):
    """Paso 3: Reescribe y optimiza el CV completo para ATS."""
    prompt_reescritura = f"""
    Actúa como un Redactor Profesional de Currículums especializado en optimización para filtros ATS.
    Toma el siguiente Currículum Vitae original y reescríbelo por completo para maximizar su afinidad con la Descripción del Trabajo proporcionada.

    Reglas estrictas:
    - Mantén estrictamente la veracidad de la experiencia y formación del candidato (no inventes empleos ni títulos).
    - Integra de forma natural las palabras clave relevantes de la oferta laboral.
    - Reescribe las responsabilidades usando verbos de acción de alto impacto y enfoque en logros.
    - Estructura el CV de manera limpia y profesional (Perfil Profesional, Experiencia Laboral, Educación, Habilidades Técnicas y Blandas).
    - Devuelve únicamente el texto del CV reforzado listo para colocar en un documento, sin saludos ni comentarios adicionales.

    ---
    DESCRIPCIÓN DEL TRABAJO:
    {texto_oferta}
    ---
    CURRÍCULUM VITAE ORIGINAL:
    {texto_cv}
    ---
    """
    return llamar_gemini_con_respaldo(prompt_reescritura, modelo)


def generar_archivo_docx(texto_cv_mejorado):
    """Convierte el texto del CV mejorado en un archivo Word (.docx) en memoria."""
    doc = Document()
    doc.add_heading("Currículum Vitae Optimizado", level=1)
    for linea in texto_cv_mejorado.split("\n"):
        linea_limpia = linea.strip()
        if not linea_limpia:
            continue
        if linea_limpia.startswith("# "):
            doc.add_heading(linea_limpia.replace("# ", ""), level=1)
        elif linea_limpia.startswith("## "):
            doc.add_heading(linea_limpia.replace("## ", ""), level=2)
        elif linea_limpia.startswith("### "):
            doc.add_heading(linea_limpia.replace("### ", ""), level=3)
        elif linea_limpia.startswith("- ") or linea_limpia.startswith("* "):
            doc.add_paragraph(linea_limpia[2:], style="List Bullet")
        else:
            doc.add_paragraph(linea_limpia.replace("**", ""))

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def registrar_descarga():
    """Callback que se ejecuta en el instante en que el usuario hace clic en descargar."""
    st.session_state.descarga_realizada = True
    st.session_state.timestamp_descarga = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # -------------------------------------------------------------------------
    # PUNTO DE CONEXIÓN PARA SUPABASE (Métrica de conversión/descarga):
    # Aquí podrás ejecutar tu UPDATE o INSERT hacia Supabase usando:
    # - descarga_realizada: True
    # - timestamp_descarga: st.session_state.timestamp_descarga
    # -------------------------------------------------------------------------


# --- Interfaz de Usuario en Streamlit ---
st.title("🚀 Optimiza tu CV con Inteligencia Artificial y Minería de Texto")
st.markdown(
    "Sube tu CV en formato PDF y pega la oferta laboral. Los resultados aparecerán **paso a paso** para agilizar tu experiencia."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Tu Currículum")
    archivo_pdf = st.file_uploader("Sube tu CV (PDF)", type=["pdf"])

with col2:
    st.subheader("2. Oferta de Empleo")
    descripcion_trabajo = st.text_area(
        "Pega aquí la descripción del puesto de trabajo", height=200
    )

if st.button("Iniciar Evaluación Progresiva", type="primary"):
    if archivo_pdf is not None and descripcion_trabajo.strip() != "":
        texto_extraido = extraer_texto_pdf(archivo_pdf)
        if not texto_extraido.strip():
            st.warning("El PDF parece estar vacío o protegido. Intenta con otro archivo.")
        else:
            # Reiniciar estado para nuevo análisis y ejecutar Paso 1 al instante
            st.session_state.texto_cv = texto_extraido
            st.session_state.descripcion_trabajo = descripcion_trabajo
            st.session_state.porcentaje_afinidad = calcular_similitud_tfidf(
                texto_extraido, descripcion_trabajo
            )
            st.session_state.feedback_ia = ""
            st.session_state.cv_mejorado = ""
            st.session_state.descarga_realizada = False
            st.session_state.timestamp_descarga = None
            st.session_state.etapa_completada = 1
    else:
        st.warning("⚠️ Por favor, sube un archivo PDF y escribe la descripción del trabajo antes de iniciar.")

# --- Renderizado Progresivo por Etapas ---
if st.session_state.etapa_completada >= 1:
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")

    # ETAPA 1: Porcentaje de Afinidad (Se muestra de inmediato en < 1 segundo)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="🎯 Paso 1: Porcentaje de Afinidad (Similitud Léxica TF-IDF)",
            value=f"{st.session_state.porcentaje_afinidad}%",
        )
    with col_m2:
        estado_descarga = (
            "✅ Descargado"
            if st.session_state.descarga_realizada
            else "⏳ Pendiente de descarga"
        )
        st.metric(
            label="📥 Estado del CV Reforzado (Métrica Supabase)",
            value=estado_descarga,
        )

    # ETAPA 2: Observaciones de mejora de Gemini AI
    st.markdown("### 💡 Paso 2: Observaciones y Recomendaciones de Gemini AI")
    contenedor_diagnostico = st.empty()

    if st.session_state.etapa_completada == 1:
        with contenedor_diagnostico.container():
            with st.spinner("Analizando oportunidades de mejora con Gemini AI..."):
                try:
                    diagnostico = generar_diagnostico_ia(
                        st.session_state.texto_cv,
                        st.session_state.descripcion_trabajo,
                        modelo_gemini,
                    )
                    st.session_state.feedback_ia = diagnostico
                    st.session_state.etapa_completada = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al generar observaciones con Gemini: {e}")
    else:
        contenedor_diagnostico.markdown(st.session_state.feedback_ia)

# ETAPA 3: Generación y descarga del CV Reforzado (se ejecuta solo tras mostrar Etapa 2)
if st.session_state.etapa_completada >= 2:
    st.markdown("---")
    st.markdown("### 📄 Paso 3: Tu CV Reforzado (Optimizado para ATS)")
    contenedor_cv = st.empty()

    if st.session_state.etapa_completada == 2:
        with contenedor_cv.container():
            with st.spinner("Redactando tu CV reforzado y preparando descarga en Word..."):
                try:
                    cv_opt = generar_cv_reforzado_ia(
                        st.session_state.texto_cv,
                        st.session_state.descripcion_trabajo,
                        modelo_gemini,
                    )
                    st.session_state.cv_mejorado = cv_opt
                    st.session_state.etapa_completada = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al redactar el CV reforzado: {e}")
    else:
        with contenedor_cv.container():
            st.info(
                "Este borrador integra las palabras clave de la oferta y verbos de impacto manteniendo tu experiencia real."
            )
            st.text_area(
                "Vista previa del contenido optimizado:",
                value=st.session_state.cv_mejorado,
                height=350,
            )

            archivo_docx = generar_archivo_docx(st.session_state.cv_mejorado)

            st.download_button(
                label="📥 Descargar CV Reforzado en Word (.docx)",
                data=archivo_docx,
                file_name="CV_Reforzado_Optimizado.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                on_click=registrar_descarga,
            )

            if st.session_state.descarga_realizada:
                st.success(
                    f"Evento registrado para analítica: descarga confirmada a las `{st.session_state.timestamp_descarga}`."
                )
