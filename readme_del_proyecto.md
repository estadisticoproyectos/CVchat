# 🚀 Analizador y Optimizador de CV con IA y Minería de Texto

Aplicación web interactiva desarrollada en **Python** y **Streamlit** que permite a los candidatos evaluar la afinidad de su Currículum Vitae (en formato PDF) frente a una oferta laboral específica, combinando técnicas de **minería de texto** (TF-IDF y Similitud de Coseno) con inteligencia artificial generativa (**Google Gemini**).

## ✨ Características Principales

* **Extracción Automática de Texto:** Lectura rápida y limpia de archivos PDF mediante `PyMuPDF`.
* **Análisis Cuantitativo (Minería de Texto):** Cálculo inmediato del índice de similitud léxica mediante vectorización TF-IDF y similitud de coseno (`scikit-learn`).
* **Análisis Cualitativo (IA Generativa):** Evaluación profunda con rol de reclutador técnico usando la API de Gemini (`google-genai`) para detectar palabras clave faltantes, sugerir verbos de impacto y proponer mejoras estratégicas.
* **Preparado para Analítica:** Estructura lista para integrar registro de métricas de uso en **Supabase**.

## 🛠️ Stack Tecnológico

* **Frontend / Interfaz:** Streamlit
* **Lenguaje:** Python 3.9+
* **Minería de Texto & Datos:** Scikit-Learn, Pandas, PyMuPDF (`fitz`)
* **Modelo de Lenguaje (LLM):** Google GenAI SDK (Gemini)
* **Base de Datos (Opcional):** Supabase

## 🚀 Instalación y Uso Local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/analizador-cv-ia.git
   cd analizador-cv-ia
   ```

2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la API Key:**
   Crea una carpeta `.streamlit` en la raíz del proyecto y dentro un archivo `secrets.toml` con tu clave de Google AI Studio:
   ```toml
   GEMINI_API_KEY = "tu_api_key_aqui"
   ```

4. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube este repositorio a tu cuenta de GitHub.
2. Conecta el repositorio en [share.streamlit.io](https://share.streamlit.io/).
3. En la configuración avanzada (**Secrets**), pega tu variable `GEMINI_API_KEY`.