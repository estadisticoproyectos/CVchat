import os
import streamlit as st
import fitz  # PyMuPDF para leer PDFs
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Analizador y Optimizador de CV con IA", page_icon="📄", layout="wide"
)

# Inicializar cliente de Google GenAI
# Asegúrate de configurar tu API Key en los secretos de Streamlit o variables de entorno
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not api_key:
    st.error(
        "⚠️ Falta la API Key de Gemini. Configúrala en los Secrets de Streamlit."
    )
    st.stop()

client = genai.Client(api_key=api_key)


def extraer_texto_pdf(uploaded_file):
    """Extrae texto de un archivo PDF subido usando PyMuPDF."""
    texto = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def calcular_similitud_tfidf(texto_cv, texto_oferta):
    """Calcula la similitud de Coseno básica entre el CV y la oferta laboral."""
    vectorizer = TfidfVectorizer(stop_words="spanish")
    tfidf_matrix = vectorizer.fit_transform([texto_cv, texto_oferta])
    similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similitud * 100, 2)


def analizar_cv_con_ia(texto_cv, texto_oferta):
    """Utiliza Gemini para generar recomendaciones cualitativas y de mejora."""
    prompt = f"""
    Actúa como un Reclutador Técnico experto y Especialista en Recursos Humanos.
    Analiza el siguiente Currículum Vitae en relación con la Descripción del Trabajo proporcionada.

    ---
    DESCRIPCIÓN DEL TRABAJO:
    {texto_oferta}
    ---
    CURRÍCULUM VITAE:
    {texto_cv}
    ---

    Por favor, entrega un análisis estructurado con los siguientes puntos:
    1. **Afinidad General y Diagnóstico:** Un breve resumen de qué tan bien encaja el perfil.
    2. **Palabras Clave Faltantes (Hard/Soft Skills):** Qué tecnologías, herramientas o conceptos clave de la oferta no se detectan en el CV.
    3. **Mejoras de Redacción y Verbos de Acción:** Sugerencias puntuales para transformar viñetas pasivas en logros orientados a resultados usando verbos de impacto.
    4. **Ideas Clave a Añadir:** Qué proyectos, certificaciones o enfoque debería destacar más el candidato.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error al conectar con la API de Gemini: {e}"


# --- Interfaz de Usuario en Streamlit ---
st.title("🚀 Optimiza tu CV con Inteligencia Artificial y Minería de Texto")
st.markdown(
    "Sube tu CV en formato PDF y pega la oferta laboral para evaluar tu afinidad y recibir recomendaciones de mejora."
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

if st.button("Analizar y Optimizar CV", type="primary"):
    if archivo_pdf is not None and descripcion_trabajo.strip() != "":
        with st.spinner(
            "Extrayendo texto y procesando análisis con IA..."
        ):
            # Extraer texto del PDF
            texto_cv = extraer_texto_pdf(archivo_pdf)

            if not texto_cv.strip():
                st.warning(
                    "El PDF parece estar vacío o protegido. Intenta con otro archivo."
                )
            else:
                # 1. Minería de texto / Métrica cuantitativa
                porcentaje_afinidad = calcular_similitud_tfidf(
                    texto_cv, descripcion_trabajo
                )

                st.markdown("---")
                st.subheader("📊 Resultados del Análisis")

                # Métrica visual
                st.metric(
                    label="Índice de Similitud Léxica (Afinidad Base)",
                    value=f"{porcentaje_afinidad}%",
                )

                # 2. Análisis cualitativo con IA
                with st.spinner("Generando recomendaciones detalladas..."):
                    feedback_ia = analizar_cv_con_ia(texto_cv, descripcion_trabajo)

                st.markdown("### 💡 Recomendaciones del Reclutador IA")
                st.write(feedback_ia)
    else:
        st.warning(
            "⚠️ Por favor, sube un archivo PDF y escribe la descripción del trabajo antes de analizar."
        )
