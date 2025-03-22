# Archivo `embedded_data.py`

Este archivo funciona como último respaldo para la base de datos vectorial.

## Propósito

El archivo `embedded_data.py` contiene una constante Python `EMBEDDED_DATABASE` que incluye todos los registros de la base de datos vectorial integrados directamente en el código. Este enfoque garantiza que la aplicación tenga una copia funcional de la base de datos aun cuando los archivos JSON y pickle no estén disponibles.

## Instrucciones para subir el archivo

Debido al gran tamaño de este archivo, es necesario subirlo manualmente:

1. Navegue a la raíz de su repositorio GitHub
2. Haga clic en "Add file" > "Upload files"
3. Arrastre o seleccione el archivo `embedded_data.py` desde su carpeta local `C:\Users\Juan\Downloads\docling\`
4. Confirme los cambios con un mensaje descriptivo

## Estructura del archivo

El archivo tiene un formato similar a este:

```python
# This file was auto-generated
# It contains the embedded vector database as a Python constant
# -*- coding: utf-8 -*-

EMBEDDED_DATABASE = [
    {
        'text': 'Contenido del documento...',
        'vector': [0.123, 0.456, ...],  # Vector de 1536 dimensiones
        'filename': 'nombre_archivo.pdf',
        'title': 'Título del documento',
        'page_numbers_str': '[10, 11]'
    },
    # Más registros aquí...
]
```

## Información importante

Este archivo es opcional pero recomendado como respaldo. La aplicación intentará cargar la base de datos en el siguiente orden:
1. `data/embedded_database.json`
2. `data/embedded_database.pkl`
3. `embedded_data.py` (este archivo)
