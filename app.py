import streamlit as st
import numpy as np
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

def clear_chat_history():
    st.session_state["messages"]=[{'role': 'assistant', 'content': "¡Hola! soy el asistente virtual de IMF, ¿en qué te puedo ayudar?"}]
    st.session_state["fuentes"]=[]
    memory.clear()


def obtener_doc_sources(respuesta):

    lista_docs = respuesta['source_documents']
    documentos = ["Página: " + str(elemento.metadata['page']) + " - Contenido: " + elemento.page_content for elemento in lista_docs]

    return documentos
    
if "messages" not in st.session_state:
    st.session_state["messages"]=[{'role': 'assistant', 'content': "¡Hola! soy el asistente virtual de IMF, ¿en qué te puedo ayudar?"}]
    st.session_state["fuentes"]=[]

@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question")
    return memory

memory = init_memory()

with st.sidebar:

    st.sidebar.button('Limpiar conversación', on_click=clear_chat_history) # si pulsas, se resetea el session state

    st.sidebar.markdown("## Opciones del chat")
    
    cbox_fuentes = st.checkbox('Mostrar fuentes')

    st.sidebar.markdown("## Configuración del modelo")

    numerodocumentos = st.sidebar.number_input('Documentos a recuperar',value=3, 
                                               min_value = 1, max_value = 5)
    
    temperature = st.slider(
    label="Temperatura",
    min_value=0.0,
    max_value=1.0,
    value=0.5)
    

# Cargamos el modelo de embedding
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

# Cargamos el vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_CHATBOT"],
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    embedding=huggingface_embeddings,
)

# Cargamos el llm

llm = HuggingFaceEndpoint(    
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.2", #"meta-llama/Llama-3.2-1B", #"meta-llama/Llama-3.2-3B-Instruct"
    temperature = 0.5, # a mayor temperatura, mayor creatividad y menos conservador
    model_kwargs={"max_length":2048})
                  #"max_new_tokens": 500})


# Crear el retriever a partir del VectorStore
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": numerodocumentos} # Consultas basadas en los 3 mejores resultados
    )


for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar = "👩‍💻").write(msg["content"])
    
user_question = st.chat_input()

if user_question:

    prompt_template = """Eres un comercial especializado de una escuela de negocios que asesora a futuros alumnos sobre másters.
    Contesta la pregunta basandote en el contexto (delimitado por <ctx> </ctx>) y en el histórico del chat (delimitado por <hs></hs>) de abajo.
    1. Da una respuesta lo más concisa posible y útil
    2. Si la respuesta no aparece en el contexto proporcionado di que no tienes la información.

    Información proporcionada
    -------
    <ctx>
    Contexto: {context}
    </ctx>
    -------
    <hs>
    Histórico: {chat_history}
    </hs>
    -------
    Pregunta: {question}
    Respuesta útil:
    """

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question",  "chat_history"])

    st.session_state["messages"].append({'role': 'user', 'content': user_question}) # añadimos la pregunta al session state 
    st.chat_message("user", avatar = "👩‍💻").markdown(user_question) # la escribimos

    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT,
                           "memory": memory})
    
    respuesta = retrievalQA.invoke({"query": user_question})

    st.session_state["messages"].append({'role': 'assistant', 'content': respuesta['result']}) # guardamos la respuesta en el session state
    st.chat_message("assistant", avatar = "👩‍💻").write(str(respuesta['result'])) # la escribimos

    fuente = obtener_doc_sources(respuesta)
    st.session_state["fuentes"].append(fuente)
    
if cbox_fuentes and len(st.session_state["fuentes"])>0:
    st.write("Fuentes: ", st.session_state["fuentes"][-1])