import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")

# --- LÓGICA DE IA ---
@st.cache_resource
def configurar_asistente():
    # 1. Cargar PDFs
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Partir textos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Embeddings y Vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. Modelo Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    # 5. El Prompt Académico UCV
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un profesor del diplomado de educación terapéutica en diabetes de la UCV. "
                   "Guía sobre conocimiento y autoeficacia usando Bandura y Carga Cognitiva. "
                   "Responde EXCLUSIVAMENTE basándote en este contexto: {context}"),
        ("human", "{input}")
    ])
    
    # 6. Crear la cadena de respuesta
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Inicializar asistente
try:
    asistente = configurar_asistente()
except Exception as e:
    st.error(f"Error al iniciar el sistema: {e}")
    st.stop()

# --- INTERFAZ ---
with st.sidebar:
    st.title("Diplomado UCV")
    st.write("Expertos en Educación en Diabetes")
    if st.button("Reiniciar Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🩺 Tutor Inteligente de Diabetes")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if duda := st.chat_input("¿Cómo aplicar la teoría de Bandura?"):
    st.session_state.messages.append({"role": "user", "content": duda})
    with st.chat_message("user"):
        st.markdown(duda)
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando bases teóricas..."):
            res = asistente.invoke({"input": duda})
            respuesta = res["answer"]
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
