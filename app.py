import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="Asistente Diabetes", page_icon="🩺")

# Barra lateral (Sidebar)
with st.sidebar:
    st.title("Educación en Diabetes")
    st.markdown("---")
    st.info("Este asistente responde basado en tus guías de automanejo.")
    
    st.subheader("Consultas Rápidas")
    if st.button("Metas de Glucemia"):
        st.session_state.prompt_rápido = "¿Cuáles son las metas de glucemia recomendadas?"
    if st.button("Signos de Hipoglucemia"):
        st.session_state.prompt_rápido = "¿Qué síntomas tiene la hipoglucemia?"

# --- LÓGICA DE INTELIGENCIA ARTIFICIAL ---
@st.cache_resource
def preparar_asistente():
    # Carga PDFs de la carpeta 'documentos' en tu GitHub
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Usa la clave de OpenAI guardada en los Secrets de Streamlit
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Inicializar el asistente
asistente = preparar_asistente()

# --- CHAT ---
st.title("🩺 Asistente para Educadores")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada de texto (manual o por botones)
input_usuario = st.chat_input("Haz una pregunta sobre el manejo de la diabetes...")
prompt = input_usuario or st.session_state.pop("prompt_rápido", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Revisando manuales..."):
            respuesta = asistente.run(prompt)
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
