
import streamlit as st
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain


load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

def clear_chat_history():
    st.session_state["messages"]=[{'role': 'assistant', 'content': "¡Hola! soy el asistente virtual, ¿en qué te puedo ayudar?"}]
    st.session_state["fuentes"]=[]
    st.session_state["chat_history"]=[]

def obtener_doc_sources(lista):
    
    return [elemento.page_content for elemento in lista]
    
if "messages" not in st.session_state:
    st.session_state["messages"]=[{'role': 'assistant', 'content': "¡Hola! soy el asistente virtual, ¿en qué te puedo ayudar?"}]
    st.session_state["fuentes"]=[]
    st.session_state["chat_history"]=[]


with st.sidebar:
    st.sidebar.button('Limpiar conversación', on_click=clear_chat_history)
    st.sidebar.markdown("## Opciones del chat")
    
    cbox_fuentes = st.checkbox('Mostrar fuentes')


    numerodocumentos = st.sidebar.number_input('Documentos a recuperar',value=4)

    cbox_memoria = st.checkbox('Mostrar memoria')
        
    numMensajes = st.slider(
        label="Mensajes anteriores incluidos en la memoria",
        min_value=1,
        max_value=10,
        value=3)

    st.sidebar.markdown("## Configuración del modelo")

    max_tokens = st.slider(
    label="Respuesta máxima",
    min_value=1,
    max_value=2000,
    value=800)
    
    temperature = st.slider(
    label="Temperatura",
    min_value=0.0,
    max_value=1.0,
    value=0.7)
    

llm = HuggingFaceHub(    
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature":temperature, "max_length":5000, "max_new_tokens": max_tokens})

@st.cache_resource
def init_memory(numMensajes):
    return ConversationBufferWindowMemory(
        llm=llm,
        input_key="question",
        output_key='answer',
        memory_key='chat_history',
        k=5,
        return_messages=True)

memory = init_memory(numMensajes)


for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar = "👩‍💻").write(msg["content"])
    
user_question = st.chat_input()




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


# Crear el retriever a partir del VectorStore
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3} # Consultas basadas en los 3 mejores resultados
    )


if user_question:

    prompt_template = """Eres un comercial especializado de una escuela de negocios que asesora a futuros alumnos sobre másters.
    Contesta la pregunta basandote en el contexto (delimitado por <ctx> </ctx>) y en el histórico del chat (delimitado por <hs></hs>) de abajo.
    1. Da una respuesta lo más concisa posible.
    2. Si no sabes la respuesta, no intentes inventarla, simplemente di que no tienes la información.
    3. Limítate a responder a la pregunta.

    Información proporcionada
    -------
    <ctx>
    {context}
    </ctx>
    -------
    <hs>
    {chat_history}
    </hs>
    -------
    Pregunta: {question}
    Respuesta útil:
    """

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
    )

    st.session_state["messages"].append({'role': 'user', 'content': user_question})
    st.chat_message("user", avatar = "👩‍💻").write(user_question)



    qa = ConversationalRetrievalChain.from_llm(llm, chain_type="stuff", 
                                retriever=retriever, 
                                return_source_documents=True,
                                verbose = True,
                                combine_docs_chain_kwargs={'prompt': PROMPT},
                                memory = memory,
                                return_generated_question = False,
                                )
    
    respuesta = qa({"question": user_question})

    st.session_state["messages"].append({'role': 'assistant', 'content': respuesta['answer']})
    st.chat_message("assistant").write(respuesta['answer'])
    
    fuente = obtener_doc_sources(respuesta["source_documents"])
    st.session_state["fuentes"].append(fuente)

    st.session_state["chat_history"].append(respuesta['chat_history'])

    
if cbox_fuentes and len(st.session_state["fuentes"])>0:
    st.write("Fuentes: ", st.session_state["fuentes"][-1])
    

# Crear la aplicación Streamlit
#st.title("Asesoramiento de Másteres - Asistente Virtual")
#st.write("Este es un asistente virtual que te ayuda a decidir cuál de nuestros másteres en Big Data, Inteligencia Artificial y Data Science es el más adecuado para ti.")

# Mantener el historial de la conversación
#if "history" not in st.session_state:
 #   st.session_state["history"] = []

# Input del usuario
#query = st.text_input("Escribe tu pregunta aquí:", key="question")

# Botón para enviar la consulta
#if st.button("Enviar"):
   # if query:
        # Agregar la pregunta al historial
      #  st.session_state.history.append({"role": "user", "content": query})

        # Recuperar documentos relevantes usando el retriever
      #  docs = retriever.get_relevant_documents(query)
      #  context = "\n\n".join([doc.page_content for doc in docs])

        # Obtener la respuesta del modelo usando invoke()
     #   result = qa.invoke({"question": query})

        # try:
        #     result = retrievalQA.invoke({
        #         "query": query,
        #         "context": context
        #     })

        # except Exception as e:
        #     st.error(f"Ocurrió un error al intentar obtener la respuesta: {str(e)}")
        #     result = None

        # Procesar y mostrar la respuesta si es posible
       # if result:
          #  st.write(result['answer'])
            #answer = result.get('result', "No se pudo obtener una respuesta.")
          #  st.session_state.history.append({"role": "agent", "content": result['answer']})

# Mostrar el historial de la conversación en la pantalla
#for msg in st.session_state.history:
    #if msg["role"] == "user":
   #     st.write(f"**Usuario:** {msg['content']}")
   # elif msg["role"] == "agent":
   #     st.write(f"**Agente:** {msg['content']}")
