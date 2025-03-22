# Instrucciones de Despliegue Manual

Debido al gran tamaño de los archivos de base de datos, es necesario subir manualmente algunos archivos a través de la interfaz web de GitHub.

## Pasos para completar el despliegue

1. **Subir archivos de base de datos vectorial**:
   - Navegue a la carpeta `data` en su repositorio de GitHub
   - Haga clic en "Add file" > "Upload files"
   - Suba los siguientes archivos desde su carpeta local `C:\Users\Juan\Downloads\docling\data\`:
     - `embedded_database.json`
     - `embedded_database.pkl`
   - Confirme los cambios con un mensaje como "Add embedded database files"

2. **Subir archivo de respaldo de datos embebidos**:
   - Navegue a la raíz de su repositorio en GitHub
   - Haga clic en "Add file" > "Upload files"
   - Suba el archivo `embedded_data.py` desde su carpeta local `C:\Users\Juan\Downloads\docling\`
   - Confirme los cambios con un mensaje como "Add embedded database Python backup"

3. **Crear/actualizar archivo `.env.example`**:
   - Navegue a la raíz de su repositorio en GitHub
   - Edite el archivo `.env.example` (o créelo si no existe)
   - Asegúrese de que contenga la siguiente línea:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Confirme los cambios

4. **Configurar en Streamlit Cloud**:
   - Inicie sesión en [Streamlit Cloud](https://streamlit.io/cloud)
   - Haga clic en "New app"
   - Conecte con su repositorio de GitHub
   - Configure:
     - **Main file path**: `streamlit_app.py`
     - En "Advanced settings" > "Secrets", agregue:
       ```
       OPENAI_API_KEY = "su_clave_de_api_real"
       ```
   - Haga clic en "Deploy"

## Verificar el despliegue

Una vez desplegada, confirme que:

1. La aplicación carga correctamente
2. La base de datos se carga sin errores
3. Las búsquedas funcionan correctamente
4. Los caracteres españoles se muestran adecuadamente

## Solución de problemas comunes

- **Error "Failed to load database from any source"**: Verifique que los archivos de base de datos se hayan subido correctamente a la carpeta `data/`.
- **Problemas con caracteres españoles**: Confirme que los archivos JSON y Python se hayan guardado con codificación UTF-8.
- **Errores de API de OpenAI**: Verifique que haya configurado correctamente la clave API en los secretos de Streamlit.
