import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Diplomado Diabetes UCV", page_icon="🩺")

# --- LÓGICA DE IA Y PROMPT PERSONALIZADO ---
@st.cache_resource
def preparar_asistente():
    # 1. Cargar PDFs desde la carpeta física en GitHub
    loader = PyPDFDirectoryLoader("documentos/")
    docs = loader.load()
    
    # 2. Fragmentar textos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 3. Crear embeddings y base de datos vectorial
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. DEFINICIÓN DEL PROMPT DE EXPERTO (Tu instrucción académica)
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

    # 5. Configurar la cadena de respuesta
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT_UCV}
    )

# Inicialización
asistente_ucv = preparar_asistente()

# --- INTERFAZ DE USUARIO (Frontend Streamlit) ---

with st.sidebar:
    st.image("https://wikimedia.org", width=100)
    st.title("Diplomado UCV")
    st.markdown("**Educación Terapéutica en Diabetes**")
    st.divider()
    st.write("Este asistente utiliza el marco teórico de Bandura y Carga Cognitiva.")
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
            # Aquí es donde ocurre la magia de tu prompt
            respuesta = asistente_ucv.invoke(prompt)
            texto_respuesta = respuesta["result"]
            
            st.markdown(texto_respuesta)
            st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
