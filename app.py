import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Estas son las nuevas rutas para evitar el ModuleNotFoundError:
from langchain.chains.retrieval import create_retrieval_chain

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente UCV Diabetes", page_icon="🩺")

@st.cache_resource
def preparar_asistente():
    # 1. Cargar PDFs
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Dividir texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Embeddings
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

    # 5. Definir el Prompt (Tu metodología UCV)
    system_prompt = (
        """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición. 
    Para ello, integrarás y aplicarás los principios de la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura y las herramientas de las precauciones universales de alfabetización en salud, tal como se definen en tus documentos de referencia.
    Cuando te pregunten cómo enseñar un aspecto específico o planificar una actividad, debes:
    1. Sugerir métodos didácticos adecuados y concretos.
    2. Justificar tus sugerencias explicando cómo se alinean con las bases teóricas mencionadas (carga cognitiva, autoeficacia, alfabetización en salud o neurociencia).
    3. Ofrecer ejemplos prácticos y aplicables.
    4. Enfatizar la diferencia entre 'dar información' y 'educar' terapéuticamente.
    5. Basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado por los documentos. Si la información no está en el contexto, indica claramente que no puedes responder. No inventes. "
        Basa tus respuestas EXCLUSIVAMENTE en el contexto: {context}"""
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Crear la nueva cadena de recuperación
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    
    return retrieval_chain

# Inicializar
try:
    asistente_ucv = preparar_asistente()
except Exception as e:
    st.error(f"Error al iniciar: {e}")
    st.stop()

# --- INTERFAZ ---
st.title("🩺 Tutor UCV Diabetes")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_input := st.chat_input("¿Cómo aplicar la teoría de Bandura?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Consultando guías..."):
            # Nota: Ahora se usa .invoke() y el resultado viene en 'answer'
            response = asistente_ucv.invoke({"input": user_input})
            respuesta_final = response["answer"]
            st.markdown(respuesta_final)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
