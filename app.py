import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")

# Configuración directa de Google (Evita el error 404 de LangChain)
import google.ai.generativelanguage as gapic
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def preparar_conocimiento():
    # 1. Cargar PDFs de la carpeta documentos
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Dividir texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Embeddings (Usando el que ya te funcionó)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    # 4. Crear base de datos de búsqueda
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Inicializar sistema
try:
    base_datos = preparar_conocimiento()
except Exception as e:
    st.error(f"Error al procesar PDFs: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🩺 Tutor Inteligente de Diabetes (UCV)")
st.caption("Marco Teórico: Bandura, Carga Cognitiva y Alfabetización en Salud")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada del usuario
if prompt_usuario := st.chat_input("Escribe tu duda académica..."):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    # Respuesta del Asistente
    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos de la UCV..."):
            try:
                # A. Buscar información relevante en tus PDFs
                busqueda = base_datos.similarity_search(prompt_usuario, k=3)
                contexto_pdfs = "\n\n".join([doc.page_content for doc in busqueda])
            
            # B. Crear el mensaje para Gemini
                instruccion_maestra = f"""
                Eres un profesor del diplomado de educación terapéutica en diabetes de la UCV.
                Responde usando la teoría de la carga cognitiva, Bandura y alfabetización en salud.
                Contexto: {contexto_pdfs}
                """
            
            # C. Llamada directa
                model = genai.GenerativeModel('models/gemini-1.5-flash')
                response = model.generate_content(f"{instruccion_maestra}\n\nPregunta: {prompt_usuario}")
                
                respuesta_texto = response.text
                st.markdown(respuesta_texto)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
                
            except Exception as e:
                st.error(f"Error en el motor de respuesta: {e}")
