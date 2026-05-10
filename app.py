import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")

@st.cache_resource
def configurar_asistente():
    # 1. Cargar PDFs
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Dividir texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    
    # 3. Definir Embeddings (¡Aquí está la línea que faltaba!)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 4. Crear base de datos vectorial
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 5. Configurar Modelo de Chat
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 6. Definir el Prompt Académico UCV
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres profesor de la UCV. Guía sobre conocimiento y autoeficacia usando Bandura y Carga Cognitiva. Responde solo según el contexto: {context}"),
        ("human", "{input}"),
    ])
    
    # 7. Crear la cadena
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Ejecución de la app
try:
    asistente_ucv = configurar_asistente()
except Exception as e:
    st.error(f"Error técnico al iniciar: {e}")
    if st.button("Limpiar caché y reintentar"):
        st.cache_resource.clear()
        st.rerun()
    st.stop()

st.title("🩺 Tutor Inteligente de Diabetes (UCV)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt_usuario := st.chat_input("Escribe tu duda aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    with st.chat_message("assistant"):
        response = asistente_ucv.invoke({"input": prompt_usuario})
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
