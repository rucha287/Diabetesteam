import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")

@st.cache_resource
def configurar_asistente():
    # 1. Cargar PDFs de la carpeta documentos
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Dividir texto en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Configurar Embeddings (Forzando v1 para evitar 404)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 4. Crear base de datos vectorial
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 5. Configurar Modelo de Chat (Forzando v1 para evitar 404)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 6. Definir el Prompt Académico de la UCV
    system_prompt = (
        "Eres un profesor del diplomado de educación terapéutica en diabetes de la UCV. "
        "Tu propósito es guiar sobre conocimiento y autoeficacia usando Bandura y Carga Cognitiva. "
        "Basa tus respuestas EXCLUSIVAMENTE en el contexto: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 7. Crear la cadena de recuperación
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Inicialización del asistente
try:
    asistente_ucv = configurar_asistente()
except Exception as e:
    st.error(f"Error técnico al iniciar: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🩺 Tutor Inteligente de Diabetes (UCV)")
st.caption("Basado en el diseño instruccional del Diplomado de la UCV")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial del chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada del usuario
if prompt_usuario := st.chat_input("¿Cómo enseñar el cuidado de los pies según Bandura?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    # Generar respuesta de la IA
    with st.chat_message("assistant"):
        with st.spinner("Consultando bases teóricas de la UCV..."):
            response = asistente_ucv.invoke({"input": prompt_usuario})
            respuesta_final = response["answer"]
            st.markdown(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
