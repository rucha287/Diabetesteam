import streamlit as st
import time
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
    
    # 2. Fragmentos grandes para minimizar peticiones
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    
    # 3. Configurar Embeddings (Ruta oficial simplificada)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    # 4. Creación por bloques con pausa para evitar el error 429 (Cuota)
    vectorstore = None
    batch_size = 3 
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        time.sleep(2) # Pausa de seguridad
    
    # 5. Configurar Modelo de Chat (Ruta oficial simplificada)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    # 6. Prompt Académico UCV
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres profesor del diplomado de diabetes de la UCV. Responde basándote en el contexto: {context}"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Ejecución
try:
    asistente_ucv = configurar_asistente()
except Exception as e:
    st.error(f"Error técnico: {e}")
    if st.button("Limpiar caché y reintentar"):
        st.cache_resource.clear()
        st.rerun()
    st.stop()

st.title("🩺 Tutor Diabetes UCV")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt_usuario := st.chat_input("Escribe tu pregunta académica..."):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    with st.chat_message("assistant"):
        response = asistente_ucv.invoke({"input": prompt_usuario})
        st.markdown(response["answer"])
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
