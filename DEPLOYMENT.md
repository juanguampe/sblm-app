# Guía de Despliegue

Esta guía explica cómo desplegar la aplicación tanto en Streamlit Cloud como localmente.

## Despliegue en Streamlit Cloud

1. **Crear una cuenta en Streamlit Cloud**:
   - Regístrate en https://streamlit.io/cloud

2. **Vincular tu cuenta de GitHub a Streamlit Cloud**:
   - Autoriza a Streamlit para acceder a tus repositorios

3. **Desplegar la aplicación**:
   - Selecciona el repositorio "sblm-app"
   - Selecciona la rama "main"
   - En "Main file path" escribe "streamlit_app.py"
   - Haz clic en "Deploy"

4. **Configurar los secretos de Streamlit**:
   - En Streamlit Cloud, navega a tu aplicación
   - Haz clic en el menú de tres puntos y selecciona "Settings"
   - Ve a la sección "Secrets"
   - Añade tu clave API de OpenAI:
     ```
     OPENAI_API_KEY = "tu-clave-api-aquí"
     ```
   - Guarda los cambios

5. **Nota sobre la base de datos**:
   - La versión desplegada en Streamlit Cloud mostrará un aviso de "Modo de demostración"
   - Esto es normal, ya que la base de datos completa no se puede cargar directamente en Streamlit Cloud

## Despliegue Local (con funcionalidad completa)

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/juanguampe/sblm-app.git
   cd sblm-app
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la API key de OpenAI**:
   - Crea un archivo `.env` en la raíz del proyecto
   - Añade tu clave API de OpenAI:
     ```
     OPENAI_API_KEY=tu-clave-api-aquí
     ```

4. **Preparar la base de datos**:
   - Copia la carpeta `docling.lance` a `data/lancedb/` dentro del proyecto
   - Asegúrate de que la estructura sea: `data/lancedb/docling.lance/...`

5. **Ejecutar la aplicación**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Obtener la Base de Datos

Para obtener la base de datos completa, contacta con el administrador del sistema para solicitar los archivos `docling.lance`. Estos archivos contienen los embeddings vectoriales y metadatos necesarios para el funcionamiento completo de la búsqueda.