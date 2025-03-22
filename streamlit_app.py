import streamlit as st
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import sys
import pickle

# Ensure proper encoding for Spanish characters
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Print OpenAI version (for debugging)
import openai
print(f"OpenAI Python SDK version: {openai.__version__}")

# Set up the page
st.set_page_config(
    page_title="Colegio San Bartolomé La Merced",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed back to collapsed
)

# Add Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

# Custom CSS for better typography and UI elements
st.markdown("""
<style>
    /* Improved Typography */
    html, body, [class*="st-"] {
        font-family: 'Merriweather', Georgia, serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', Georgia, serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    p {
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Chat UI Improvements */
    .stChatMessage {
        border-radius: 10px;
        padding: 12px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stChatMessage[data-testid*="user"] {
        background-color: #e5f3ff !important;
    }
    .stChatMessage[data-testid*="assistant"] {
        background-color: #f8f9fa !important;
    }
    
    /* Search Results Styling */
    .search-result {
        margin-bottom: 15px;
        border-left: 3px solid #1a4a73;
        padding-left: 15px;
    }
    .search-result summary {
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 8px;
        color: #1a4a73;
    }
    .search-result .metadata {
        font-style: italic;
        color: #6c757d;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    
    /* Better Inputs */
    .stTextInput input {
        border-radius: 6px;
        border: 1px solid #ced4da;
        padding: 10px 15px;
        font-size: 1rem;
    }
    .stTextInput input:focus {
        border-color: #1a4a73;
        box-shadow: 0 0 0 0.2rem rgba(26, 74, 115, 0.25);
    }
    
    /* Prettier buttons */
    button[kind="primary"] {
        background-color: #1a4a73;
        border-radius: 6px;
        border: none;
        padding: 6px 16px;
        font-weight: 600;
    }
    button:not([kind="primary"]) {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    
    /* Login styling */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .login-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .login-icon {
        font-size: 4rem;
        color: #1a4a73;
        margin-bottom: 1rem;
    }
    .login-form {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Password authentication function
def authenticate(password):
    """Verify if the password is correct."""
    # Try to get password from secrets first (for cloud deployment)
    correct_password = None
    try:
        correct_password = st.secrets["access_password"]
    except:
        # Fallback to environment variable (for local development)
        correct_password = os.getenv("ACCESS_PASSWORD", "SBLMIgual2025")
    
    return password == correct_password

# Initialize session state variables for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login screen if not authenticated
if not st.session_state.authenticated:
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    # Use Font Awesome for a book icon instead of an image
    st.markdown("<div class='login-logo'>", unsafe_allow_html=True)
    st.markdown("<div class='login-icon'><i class='fas fa-book-open'></i></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-bottom: 1.5rem;'>Acceso al Asistente Pedagógico</h2>", unsafe_allow_html=True)
    
    password = st.text_input("Contraseña", type="password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_button = st.button("Ingresar", use_container_width=True)
    
    if login_button:
        if authenticate(password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta. Por favor, intente nuevamente.")
    
    st.markdown("<p style='text-align: center; font-size: 0.9rem; margin-top: 2rem;'>Colegio San Bartolomé La Merced<br>© 2025 Todos los derechos reservados</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Stop rendering the rest of the app
    st.stop()

# If we get here, the user is authenticated
# Get OpenAI API key from secrets or environment variables
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except:
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("No se encontró la clave API de OpenAI. Por favor, configure la clave en los secretos de Streamlit.")
    st.stop()

# Initialize OpenAI client with explicitly passed API key
client = OpenAI(api_key=openai_api_key)

# Helper function for vector similarity
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load embedded database
@st.cache_resource
def load_embedded_database():
    """Load the embedded database from file.

    Returns:
        List of records with text, vector, and metadata
    """
    try:
        # First try to load from JSON (more portable)
        try:
            with open("data/embedded_database.json", "r", encoding="utf-8") as f:
                print("Loading database from JSON...")
                records = json.load(f)
                print(f"Loaded {len(records)} records from JSON")
                return records
        except Exception as json_error:
            print(f"Error loading from JSON: {json_error}")
            
            # Fall back to pickle if JSON fails
            try:
                with open("data/embedded_database.pkl", "rb") as f:
                    print("Loading database from pickle...")
                    records = pickle.load(f)
                    print(f"Loaded {len(records)} records from pickle")
                    return records
            except Exception as pickle_error:
                print(f"Error loading from pickle: {pickle_error}")
                
                # Try from embedded data if both external files fail
                try:
                    print("Loading from embedded data...")
                    from embedded_data import EMBEDDED_DATABASE
                    print(f"Loaded {len(EMBEDDED_DATABASE)} records from embedded data")
                    return EMBEDDED_DATABASE
                except Exception as embedded_error:
                    print(f"Error loading from embedded data: {embedded_error}")
                    raise Exception("Failed to load database from any source")
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {e}")
        return None

def get_context(query: str, database, num_results: int = 5) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        database: List of database records
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    try:
        # Generate embedding for query using OpenAI
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        # Calculate similarity scores
        results = []
        for record in database:
            similarity = cosine_similarity(query_vector, record['vector'])
            results.append({
                'text': record['text'],
                'filename': record['filename'],
                'title': record['title'],
                'page_numbers_str': record['page_numbers_str'],
                'similarity': similarity
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top results
        top_results = results[:num_results]
        
        # Store search results in session state
        st.session_state.search_results = top_results
        
        # Just log the count - no expander here
        st.info(f"Se encontraron {len(top_results)} fuentes relevantes")
        
        contexts = []

        for result in top_results:
            # Get metadata fields
            filename = result["filename"]
            title = result["title"]
            
            # Parse page numbers from string back to list
            try:
                page_numbers = json.loads(result["page_numbers_str"])
            except:
                page_numbers = []
            
            # Build source citation
            source_parts = [filename]
            if page_numbers and len(page_numbers) > 0:
                source_parts.append(f"p. {', '.join(str(p) for p in page_numbers)}")

            source = f"\nFuente: {' - '.join(source_parts)}"
            if title:
                source += f"\nTítulo: {title}"

            contexts.append(f"{result['text']}{source}")

        context_text = "\n\n".join(contexts)
        # Store context in session state
        st.session_state.context = context_text
        return context_text
    except Exception as e:
        st.error(f"Error de búsqueda: {e}")
        return f"Error al recuperar contexto: {str(e)}"

def get_chat_download_str():
    """Format chat history for download.
    
    Returns:
        str: Formatted chat history
    """
    if not st.session_state.messages:
        return "No hay mensajes para descargar."
        
    download_str = "Conversación con el Asistente del Colegio San Bartolomé La Merced\n"
    download_str += "=============================================================\n\n"
    
    for message in st.session_state.messages:
        role = "Usuario" if message["role"] == "user" else "Asistente"
        download_str += f"[{role}]\n{message['content']}\n\n"
    
    download_str += "=============================================================\n"
    download_str += f"Descargado el: {time.strftime('%d/%m/%Y %H:%M:%S')}\n"
    
    return download_str

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
    
    # First, try with primary model
    try:
        model_to_use = "gpt-4o-mini"
        print(f"Intentando usar el modelo {model_to_use}")
        
        response_stream = client.chat.completions.create(
            model=model_to_use,
            messages=api_messages,
            temperature=temperature,
            stream=True
        )
        
        response = st.write_stream(response_stream)
        return response
    except Exception as primary_error:
        st.warning(f"La llamada a la API principal falló: {primary_error}. Intentando método alternativo...")
        
        # Fallback to standard Chat Completions API
        try:
            model_to_use = "gpt-3.5-turbo"  # Fallback to a more widely available model
            print(f"Usando modelo alternativo: {model_to_use}")
            
            response_stream = client.chat.completions.create(
                model=model_to_use,
                messages=api_messages,
                temperature=temperature,
                stream=True
            )
            
            response = st.write_stream(response_stream)
            return response
        except Exception as fallback_error:
            st.error(f"La llamada a la API alternativa también falló: {fallback_error}")
            return "Encontré un error al intentar generar una respuesta. Por favor, intenta de nuevo más tarde."

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

# Load database
database = load_embedded_database()

# Main content area
st.markdown("<h1 style='text-align: center; color: #1a4a73; margin-bottom: 0;'>📚 Colegio San Bartolomé La Merced</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 1.2rem; margin-bottom: 2rem;'>Realice consultas sobre los documentos en nuestra base de conocimiento.<br>Proporcionaré respuestas con fuentes relevantes.</p>", unsafe_allow_html=True)

# Styled sidebar with improved controls
with st.sidebar:
    st.markdown("<h1 style='color: #1a4a73; font-size: 1.5rem;'>Configuración</h1>", unsafe_allow_html=True)
    
    # Clear buttons with styling
    st.markdown("<h2 style='color: #1a4a73; font-size: 1.2rem; margin-top: 1.5rem;'>Acciones</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Limpiar Fuentes", use_container_width=True):
            st.session_state.context = ""
            st.session_state.search_results = []
            st.session_state.context_expanded = False
            st.rerun()
    
    # Advanced settings with improved styling
    st.markdown("<h2 style='color: #1a4a73; font-size: 1.2rem; margin-top: 2rem;'>Configuración Avanzada</h2>", unsafe_allow_html=True)
    
    # Add a divider
    st.markdown("<hr style='margin: 1rem 0; border-color: #dee2e6;'>", unsafe_allow_html=True)
    
    # Temperature with improved labels
    st.markdown("<p style='margin-bottom: 0.2rem; font-weight: 600; color: #495057;'>Creatividad de respuestas</p>", unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "Temperatura", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.temperature,
        step=0.1,
        help="Valores más altos producen respuestas más creativas, valores más bajos producen respuestas más deterministas",
        label_visibility="collapsed"
    )
    
    # Source count with descriptive labels
    st.markdown("<p style='margin: 1rem 0 0.2rem; font-weight: 600; color: #495057;'>Fuentes a recuperar</p>", unsafe_allow_html=True)
    st.session_state.results_count = st.slider(
        "Fuentes", 
        min_value=1, 
        max_value=15, 
        value=st.session_state.results_count,
        help="Más fuentes proporcionan más contexto pero pueden diluir la relevancia",
        label_visibility="collapsed"
    )
    
    # Download chat option
    st.markdown("<p style='margin: 1rem 0 0.2rem; font-weight: 600; color: #495057;'>Descargar conversación</p>", unsafe_allow_html=True)
    if st.download_button(
        label="Descargar Chat",
        data=get_chat_download_str(),
        file_name="conversacion.txt",
        mime="text/plain",
        use_container_width=True
    ):
        st.success("Conversación descargada con éxito")
    
    # Add visual indicator for database status
    st.markdown("<hr style='margin: 1.5rem 0 1rem; border-color: #dee2e6;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #d4edda; border-radius: 4px; padding: 0.75rem; font-size: 0.9rem;'><strong>Estado:</strong> Base de datos cargada<br><span style='color: #5a6268;'>{len(database)} documentos indexados</span></div>", unsafe_allow_html=True)
    
    # Logout option
    st.markdown("<hr style='margin: 1.5rem 0 1rem; border-color: #dee2e6;'>", unsafe_allow_html=True)
    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Stop if database is None
if database is None:
    st.error("Base de datos no encontrada o no inicializada correctamente. Por favor, contacte al administrador.")
    st.stop()

# Display existing chat history first
chat_column, sources_column = st.columns([3, 1])

with chat_column:
    # Display all existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Sources are shown in the right column with enhanced styling
with sources_column:
    if st.session_state.context and st.session_state.context_expanded:
        st.markdown("<h3 style='color: #1a4a73; border-bottom: 2px solid #dee2e6; padding-bottom: 8px; margin-bottom: 16px;'>Fuentes</h3>", unsafe_allow_html=True)
        
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
                        <summary><i class="fas fa-file-alt"></i> {source}</summary>
                        <div class="metadata"><strong>Sección:</strong> {title}</div>
                        <div style="margin-top: 12px; background-color: #f8f9fa; padding: 12px; border-radius: 6px; font-size: 0.95rem;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Chat input at the bottom
if prompt := st.chat_input("Escribe aquí tu pregunta..."):
    # Get relevant context first
    with st.status("Buscando información relevante...", expanded=False) as status:
        context = get_context(prompt, database, num_results=st.session_state.results_count)
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