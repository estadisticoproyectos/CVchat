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

if not api_key:
    st.error("⚠️ Falta configurar `GEMINI_API_KEY` en los Secrets de Streamlit.")
    st.stop()

if not modelo_gemini:
    st.error("⚠️ Falta configurar `GEMINI_MODEL` en los Secrets de Streamlit (ej. `gemini-3.8-flash`).")
    st.stop()

client = genai.Client(api_key=api_key)

# Inicializar estado de sesión (Session State) para persistencia y métricas
if "analisis_realizado" not in st.session_state:
    st.session_state.analisis_realizado = False
if "porcentaje_afinidad" not in st.session_state:
    st.session_state.porcentaje_afinidad = 0.0
if "feedback_ia" not in st.session_state:
    st.session_state.feedback_ia = ""
if "cv_mejorado" not in st.session_state:
    st.session_state.cv_mejorado = ""
if "descarga_realizada" not in st.session_state:
    st.session_state.descarga_realizada = False
if "timestamp_descarga" not in st.session_state:
    st.session_state.timestamp_descarga = None


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


def analizar_y_mejorar_cv(texto_cv, texto_oferta, modelo):
    """Genera diagnóstico de mejoras y una versión reescrita y reforzada del CV."""
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

    resp_diag = client.models.generate_content(model=modelo, contents=prompt_diagnostico)
    resp_cv = client.models.generate_content(model=modelo, contents=prompt_reescritura)
    return resp_diag.text, resp_cv.text


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
    "Sube tu CV en formato PDF, pega la oferta laboral, evalúa tu afinidad y descarga una versión reforzada lista para postular."
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

if st.button("Analizar y Generar CV Reforzado", type="primary"):
    if archivo_pdf is not None and descripcion_trabajo.strip() != "":
        with st.spinner("Extrayendo texto, calculando afinidad y redactando CV reforzado..."):
            texto_cv = extraer_texto_pdf(archivo_pdf)

            if not texto_cv.strip():
                st.warning("El PDF parece estar vacío o protegido. Intenta con otro archivo.")
            else:
                try:
                    afinidad = calcular_similitud_tfidf(texto_cv, descripcion_trabajo)
                    diagnostico, cv_opt = analizar_y_mejorar_cv(
                        texto_cv, descripcion_trabajo, modelo_gemini
                    )

                    # Guardar en session_state para persistir tras recargas de botones
                    st.session_state.analisis_realizado = True
                    st.session_state.porcentaje_afinidad = afinidad
                    st.session_state.feedback_ia = diagnostico
                    st.session_state.cv_mejorado = cv_opt
                    st.session_state.descarga_realizada = False
                    st.session_state.timestamp_descarga = None

                except Exception as e:
                    st.error(f"Error al procesar con la API de Gemini: {e}")
    else:
        st.warning("⚠️ Por favor, sube un archivo PDF y escribe la descripción del trabajo antes de analizar.")

# --- Mostrar Resultados Persistentes ---
if st.session_state.analisis_realizado:
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(
            label="Índice de Similitud Léxica (Afinidad Base)",
            value=f"{st.session_state.porcentaje_afinidad}%",
        )
    with col_m2:
        estado_descarga = "✅ Descargado" if st.session_state.descarga_realizada else "⏳ Pendiente de descarga"
        st.metric(
            label="Métrica de Conversión (Seguimiento de Descarga)",
            value=estado_descarga,
        )

    tab1, tab2 = st.tabs(["💡 Diagnóstico y Mejoras", "📄 CV Reforzado (Listo para Descargar)"])

    with tab1:
        st.markdown(st.session_state.feedback_ia)

    with tab2:
        st.info("Este borrador integra las palabras clave de la oferta y verbos de impacto manteniendo tu experiencia real.")
        st.text_area(
            "Vista previa del CV Reforzado:",
            value=st.session_state.cv_mejorado,
            height=400,
        )

        # Preparar archivo Word (.docx)
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
