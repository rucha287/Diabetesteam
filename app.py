import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

st.set_page_config(page_title="Tutor Diabetes UCV", page_icon="🩺")
import time # Ensure this line is at the beginning of your file

@st.cache_resource
def configurar_asistente():
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Much larger chunks to minimize requests
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents(docs)
    
    # 3. Configure Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 4. Process in batches with pause (Solution to error 429)
    # We divide the fragments into groups of 5 and wait 1 second between them
    vectorstore = None
    batch_size = 5
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings_model)
        else:
            vectorstore.add_documents(batch)
        time.sleep(1.5) # Technical pause to avoid exhausting the quota
    
    # 5. Configure Chat Model
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"],
        client_options={"api_version": "v1"}
    )
    
    # 6. Prompt and Chain (Same as before)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professor from UCV. Respond according to the context: {context}"),
        ("human", "{input}"),
    ])
    
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
