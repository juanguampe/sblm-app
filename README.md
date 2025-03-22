# Asistente Pedagogía Ignaciana - Colegio San Bartolomé La Merced

Aplicación de chat con vectores integrados para responder consultas sobre pedagogía Ignaciana y la propuesta educativa del Colegio San Bartolomé La Merced.

## Funcionalidades

- Chat interactivo con respuestas basadas en documentos institucionales
- Búsqueda semántica en vectores integrados 
- Interfaz elegante y fácil de usar
- Citación automática de fuentes
- Soporte completo para caracteres especiales del español
- Protección por contraseña para control de acceso

## Protección de Acceso

La aplicación está protegida por contraseña para controlar el acceso y el uso del API. La contraseña por defecto es `SBLMIgual2025`, pero se recomienda cambiarla en producción usando los secretos de Streamlit.

## Despliegue en Streamlit Cloud

1. Asegúrese de que los siguientes archivos estén presentes en el repositorio:
   - `streamlit_app.py` - La aplicación principal
   - `data/embedded_database.json` - Base de datos vectorial serializada
   - `data/embedded_database.pkl` - Respaldo de la base de datos
   - `embedded_data.py` - Datos integrados en código Python (opcional)
   - `requirements.txt` - Dependencias
   - `.streamlit/config.toml` - Configuración de Streamlit

2. Configurar en Streamlit Cloud:
   - **Repository**: juanguampe/sblm-app
   - **Branch**: main
   - **Main file path**: streamlit_app.py
   - **Python version**: 3.9+
   - **Secrets**: 
     - `OPENAI_API_KEY`: su_clave_de_api
     - `access_password`: su_contraseña_personalizada (opcional, por defecto es SBLMIgual2025)

## Configuración Local

Para ejecutar localmente:

1. Clone el repositorio
2. Cree un archivo `.env` con:
   ```
   OPENAI_API_KEY=su_clave_de_api
   ACCESS_PASSWORD=su_contraseña_personalizada
   ```
3. Ejecute `streamlit run streamlit_app.py`