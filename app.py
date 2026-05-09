import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# Esta es la forma más estable de importar la cadena:
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Diplomado Diabetes UCV", page_icon="🩺")

# --- LÓGICA DE IA CON GEMINI ---
@st.cache_resource
def preparar_asistente():
    # 1. Cargar PDFs desde la carpeta física en GitHub
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Fragmentar textos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Crear embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    # 4. Crear base de datos vectorial
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 5. DEFINICIÓN DEL PROMPT DE EXPERTO UCV
    template = """Eres un profesor del diplomado de educación terapéutica en diabetes de la Universidad Central de Venezuela y un experto en diseño instruccional para pacientes. Tu propósito es guiar a los educadores en diabetes sobre la mejor manera de lograr que los pacientes adquieran conocimiento y autoeficacia en el manejo de su condición. 

    Para ello, integrarás y aplicarás los principios de la teoría de la carga cognitiva, la teoría de la autoeficacia de Bandura y las herramientas de las precauciones universales de alfabetización en salud, tal como se definen en tus documentos de referencia.

    Cuando te pregunten cómo enseñar un aspecto específico o planificar una actividad, debes:
    1. Sugerir métodos didácticos adecuados y concretos.
    2. Justificar tus sugerencias explicando cómo se alinean con las bases teóricas mencionadas (carga cognitiva, autoeficacia, alfabetización en salud o neurociencia).
    3. Ofrecer ejemplos prácticos y aplicables.
    4. Enfatizar la diferencia entre 'dar información' y 'educar' terapéuticamente.
    5. Basar todas tus respuestas EXCLUSIVAMENTE en el contexto proporcionado por los documentos. Si la información no está en el contexto, indica claramente que no puedes responder. No inventes.

    CONTEXTO DE DOCUMENTOS:
    {context}

    PREGUNTA DEL EDUCADOR:
    {question}

    RESPUESTA DEL PROFESOR UCV:"""

    PROMPT_UCV = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # 6. Configurar el modelo Gemini 1.5
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.2,
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT_UCV}
    )

# Inicialización
try:
    asistente_ucv = preparar_asistente()
except Exception as e:
    st.error(f"Error al cargar el asistente: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO (Frontend Streamlit) ---

with st.sidebar:
    st.image("https://wikimedia.org", width=100)
    st.title("Diplomado UCV")
    st.markdown("**Educación Terapéutica en Diabetes**")
    st.divider()
    if st.button("Nueva sesión de tutoría"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Tutor Inteligente de Educación en Diabetes")
st.caption("Basado en el diseño instruccional del Diplomado de la UCV")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Procesar entrada del usuario
if prompt := st.chat_input("Ej: ¿Cómo enseñar el uso del glucómetro usando Bandura?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando bases teóricas y documentos..."):
            respuesta = asistente_ucv.invoke(prompt)
            texto_respuesta = respuesta["result"]
            
            st.markdown(texto_respuesta)
            st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
