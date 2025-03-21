import streamlit as st
import lancedb
import os
import json
import time
import sys
import requests
import zipfile
import io

# Function to download and extract database files if they don't exist
def download_database_if_needed():
    db_path = "data/lancedb"
    if not os.path.exists(db_path) or not os.path.exists(f"{db_path}/docling.lance"):
        st.info("Base de datos no encontrada. Descargando archivos...")
        
        # Create directories if they don't exist
        os.makedirs(db_path, exist_ok=True)
        
        # URL to the database ZIP file in the GitHub release
        release_url = "https://github.com/juanguampe/sblm-app/releases/download/v1.0.0/lancedb.zip"
        
        try:
            # Download the ZIP file
            response = requests.get(release_url)
            if response.status_code == 200:
                # Extract the ZIP file
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall("data/")
                st.success("Base de datos descargada y extraída correctamente.")
            else:
                st.error(f"No se pudo descargar la base de datos. Código de error: {response.status_code}")
        except Exception as e:
            st.error(f"Error al descargar o extraer la base de datos: {e}")

# Download database on startup if needed
download_database_if_needed()

# Create OpenAI API functions without using the client
def create_embedding(text):
    """Create embedding using OpenAI API directly with requests."""
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("No se encontró la clave API de OpenAI en los secretos de Streamlit.")
        st.stop()
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        st.error(f"Error al crear embedding: {response.status_code}")
        st.error(response.text)
        return None

def create_chat_completion(messages, temperature=0.7, stream=True):
    """Create chat completion using OpenAI API directly with requests."""
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("No se encontró la clave API de OpenAI en los secretos de Streamlit.")
        st.stop()
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "stream": stream
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=stream
        )
        
        if response.status_code != 200:
            st.error(f"Error al crear chat completion: {response.status_code}")
            st.error(response.text)
            return None
            
        if stream:
            def generate():
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                            data = json.loads(line[6:])
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
            
            return generate()
        else:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error al crear chat completion: {e}")
        return None

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    try:
        # Check if the database directory exists
        db_path = "data/lancedb"
        if not os.path.exists(db_path):
            st.error(f"¡El directorio de la base de datos {db_path} no existe!")
            return None
            
        # Connect to the database
        db = lancedb.connect(db_path)
        
        # Check if the 'docling' table exists
        tables = db.table_names()
        if "docling" not in tables:
            st.error("¡No se encontró la tabla 'docling'!")
            return None
            
        # Open the table
        return db.open_table("docling")
    except Exception as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

def get_context(query: str, table, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    try:
        # Generate embedding for query using OpenAI
        query_vector = create_embedding(query)
        if not query_vector:
            return "Error al generar embedding para la consulta."
        
        # Search using vector name parameter
        results = table.search(query_vector, vector_column_name="vector").limit(num_results).to_pandas()
        
        # Store search results in session state
        st.session_state.search_results = results
        
        # Just log the count - no expander here
        st.info(f"Se encontraron {len(results)} fuentes relevantes")
        
        contexts = []

        for i, (_, row) in enumerate(results.iterrows()):
            # Get metadata fields
            filename = row["filename"]
            title = row["title"]
            
            # Parse page numbers from string back to list
            try:
                page_numbers = json.loads(row["page_numbers_str"])
            except:
                page_numbers = []
            
            # Build source citation
            source_parts = [filename]
            if page_numbers and len(page_numbers) > 0:
                source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

            source = f"\nFuente: {' - '.join(source_parts)}"
            if title:
                source += f"\nTítulo: {title}"

            contexts.append(f"{row['text']}{source}")

        context_text = "\n\n".join(contexts)
        # Store context in session state
        st.session_state.context = context_text
        return context_text
    except Exception as e:
        st.error(f"Error de búsqueda: {e}")
        return f"Error al recuperar contexto: {str(e)}"

def get_ai_response(messages, context: str, temperature: float = 0.7) -> str:
    """Get streaming response from OpenAI API with fallback options.

    Args:
        messages: Chat history
        context: Retrieved context from database
        temperature: Model temperature (0.0 to 1.0)

    Returns:
        str: Model's response
    """
    system_prompt = f"""# Rol
Eres un asistente experto en pedagogía Ignaciana.

# Objetivo
Tu objetivo principal es proporcionar respuestas claras, concretas y detalladas acerca de la pedagogía Ignaciana y la propuesta educativa del Colegio San Bartolomé La Merced.

# Instrucciones
1. Utiliza únicamente la información contenida en tu base de conocimiento interna.
2. Si la pregunta del usuario se relaciona con temas como Paradigma Pedagógico Ignaciano (PPI), instrumentos de la Educación Personalizada u otros conceptos clave de la pedagogía Ignaciana, recurre al documento **"Coloquios para un conocimiento práctico de la propuesta educativa de la Compañía de Jesús"**.
3. Para preguntas sobre la propuesta de innovación del Colegio San Bartolomé La Merced, consulta **"MAGIS_XXI"**.
4. Si la pregunta se relaciona con la obra **"LA PEDAGOGIA IGNACIANA Y SU CARACTER ECLECTICO"**, utiliza ese documento.
5. Cuando se trate de la pedagogía Ignaciana a la luz de desarrollos pedagógicos contemporáneos, usa el documento **"Aprender por refracción"**.
6. Para inquietudes acerca del Programa de Afectividad del Colegio San Bartolomé La Merced, acude al documento **"Afectividad"**.
7. Si el usuario solicita información que no se encuentra en los documentos disponibles, indícalo de manera clara: por ejemplo, "No dispongo de información suficiente en mi base de conocimiento interna para responder esa pregunta."

# Documentos Disponibles en la Base de Conocimiento
1. **Coloquios para un conocimiento práctico de la propuesta educativa de la Compañía de Jesús**  
   - Fundamenta la pedagogía Ignaciana, el PPI, la Educación Personalizada y otros conceptos clave.
2. **MAGIS_XXI**  
   - Propuesta de innovación educativa del Colegio San Bartolomé La Merced.
3. **LA PEDAGOGIA IGNACIANA Y SU CARACTER ECLECTICO**
   - Expone la naturaleza ecléctica de la pedagogía Ignaciana.
4. **Aprender por refracción**
   - Desarrolla el enfoque de la pedagogía Ignaciana en la actualidad con base en estudios pedagógicos contemporáneos.
5. **Afectividad**
   - Describe el Programa de Afectividad del Colegio San Bartolomé La Merced.

Contexto del documento sobre el que debes responder:
{context}
"""

    api_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add the conversation history
    for message in messages:
        api_messages.append(message)
    
    # First, try with gpt-4o-mini
    try:
        print("Intentando usar el modelo gpt-4o-mini")
        
        stream_generator = create_chat_completion(
            messages=api_messages,
            temperature=temperature,
            stream=True
        )
        
        if stream_generator:
            full_response = ""
            for chunk in stream_generator:
                full_response += chunk
                st.write(chunk, end="")
            return full_response
        else:
            raise Exception("No se pudo obtener respuesta del modelo")
    except Exception as primary_error:
        st.warning(f"La llamada a la API principal falló: {primary_error}. Intentando método alternativo...")
        
        try:
            # Try with GPT-3.5
            api_messages[0]["content"] = api_messages[0]["content"] + "\n\nUsa un lenguaje sencillo y claro."
            print("Usando modelo alternativo: gpt-3.5-turbo")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": api_messages,
                "temperature": temperature,
                "stream": True
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.text}")
                
            def generate():
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except:
                                continue
            
            full_response = ""
            for chunk in generate():
                full_response += chunk
                st.write(chunk, end="")
            return full_response
            
        except Exception as fallback_error:
            st.error(f"La llamada a la API alternativa también falló: {fallback_error}")
            return "Encontré un error al intentar generar una respuesta. Por favor, intenta de nuevo más tarde."

# Set up the page
st.set_page_config(
    page_title="Colegio San Bartolomé La Merced",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed back to collapsed
)

# Initialize session state for app context and search results
if "first_run" not in st.session_state:
    st.session_state.first_run = True
    # Initialize empty context and search results
    st.session_state.context = ""
    st.session_state.search_results = []
    st.session_state.context_expanded = False

# Initialize other session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "results_count" not in st.session_state:
    st.session_state.results_count = 5

# Initialize database
table = init_db()

# Main content area
st.title("📚 Colegio San Bartolomé La Merced")
st.write("Realice consultas sobre los documentos en nuestra base de conocimiento. Proporcionaré respuestas con fuentes relevantes.")

# Simple sidebar with minimal controls
with st.sidebar:
    st.title("Configuración")
    
    # Clear buttons
    st.header("Acciones")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Limpiar Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Limpiar Fuentes"):
            st.session_state.context = ""
            st.session_state.search_results = []
            st.session_state.context_expanded = False
            st.rerun()
    
    # Advanced settings directly in sidebar
    st.header("Configuración Avanzada")
    st.session_state.temperature = st.slider(
        "Temperatura", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.temperature,
        step=0.1,
        help="Valores más altos producen respuestas más creativas, valores más bajos producen respuestas más deterministas"
    )
    
    st.session_state.results_count = st.slider(
        "Fuentes a recuperar", 
        min_value=1, 
        max_value=15, 
        value=st.session_state.results_count,
        help="Más fuentes proporcionan más contexto pero pueden diluir la relevancia"
    )

# Stop if table is None
if table is None:
    st.error("Base de datos no encontrada o no inicializada correctamente. Por favor, contacte al administrador.")
    st.stop()

# Display existing chat history first
chat_column, sources_column = st.columns([3, 1])

with chat_column:
    # Display all existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Sources are shown in the right column
with sources_column:
    if st.session_state.context and st.session_state.context_expanded:
        st.subheader("Fuentes")
        
        for chunk in st.session_state.context.split("\n\n"):
            # Split into text and metadata parts
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {}
            for line in parts[1:]:
                if ": " in line:
                    key, value = line.split(": ", 1)  # Split on first occurrence only
                    metadata[key] = value

            source = metadata.get("Fuente", "Fuente desconocida")
            title = metadata.get("Título", "Sección sin título")

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Sección: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Chat input at the bottom
if prompt := st.chat_input("Escribe aquí tu pregunta..."):
    # Get relevant context first
    with st.status("Buscando información relevante...", expanded=False) as status:
        context = get_context(prompt, table, num_results=st.session_state.results_count)
        st.session_state.context_expanded = True
        
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_column:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Generate and display assistant response
    with chat_column:
        with st.chat_message("assistant"):
            response = get_ai_response(
                st.session_state.messages, 
                context,
                temperature=st.session_state.temperature
            )
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the UI with new sources
    st.rerun()