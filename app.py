import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Importaciones desde el paquete de compatibilidad classic
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")

@st.cache_resource
def configurar_asistente():
    # 1. Cargar PDFs
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Dividir texto
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
    
    # 5. Prompt UCV
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un profesor de la UCV experto en diabetes. Usa Bandura y Carga Cognitiva. Responde solo según el contexto: {context}"),
        ("human", "{input}")
    ])
    
    # 6. Cadena de respuesta
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

asistente = configurar_asistente()

# --- INTERFAZ ---
st.title("🩺 Tutor Inteligente de Diabetes (UCV)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if duda := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": duda})
    with st.chat_message("user"):
        st.markdown(duda)
    
    with st.chat_message("assistant"):
        res = asistente.invoke({"input": duda})
        st.markdown(res["answer"])
        st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
