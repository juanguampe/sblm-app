# Archivos de Base de Datos Vectorial

Este directorio debe contener los siguientes archivos de base de datos vectorial:

1. `embedded_database.json` - Base de datos vectorial en formato JSON
2. `embedded_database.pkl` - Respaldo de la base de datos en formato pickle

## Instrucciones para subir los archivos

Debido al gran tamaño de estos archivos, es necesario subirlos manualmente a través de la interfaz web de GitHub:

1. Navegue a la carpeta `data` en su repositorio GitHub
2. Haga clic en "Add file" > "Upload files"
3. Arrastre o seleccione los archivos desde su carpeta local `C:\Users\Juan\Downloads\docling\data\`
4. Confirme los cambios con un mensaje descriptivo

## Información importante

- Estos archivos contienen vectores de incrustaciónr (embeddings) que permiten búsquedas semánticas
- Reemplaza la funcionalidad LanceDB con un enfoque más portable
- Es necesario subir ambos archivos para garantizar la compatibilidad máxima
