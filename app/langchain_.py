from langchain_openai import ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Ensure the OpenAI API key is set as an environment variable
openai_api_key = config.OPENAI_API_KEY

# Initialize OpenAI LLM with LangChain
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")
print("Working LLM" if llm else "no LLM")

# Session history storage
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def query_llm(properties, input, session_id):
    # Turns properties into runnable list of Documents
    runnable_listings = RunnableLambda(
        lambda *args: [
            Document(page_content=listing, metadata={"source": "local"})
            for listing in properties
        ]
    )
    # Contextualize question
    contextualize_q_system_prompt = """
    Dada una historia de chat y la última pregunta del usuario,
    que podría hacer referencia al contexto en la historia del chat,
    formula una pregunta independiente que pueda entenderse
    sin la historia del chat. NO respondas la pregunta,
    solo reformúlala si es necesario y, de lo contrario, devuélvela tal como está.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=runnable_listings, prompt=contextualize_q_prompt
    )

    template = """
    Eres un asistente experto de bienes raíces. Te llamas Yobot. Usa el contexto obtenido para responder preguntas.

    Intenta generar respuestas dinámicas y personalizadas para cada pregunta. Si la pregunta es acerca de inmobiliaria, responde con información relevante sobre las propiedades.

    Si la pregunta no es acerca de inmobiliaria, recuerda tu rol al usuario y no respondas la pregunta.

    Tus tareas son:

    1. **Responder preguntas concisas:** Responde a las preguntas en **1600 caracteres o menos**. **Es crucial** que la respuesta **no** exceda este límite. Si es necesario, recorta la información para ajustarte a este límite. Si la respuesta excede el límite, corta la información menos relevante.
    2. **SIEMPRE** proporciona el ID de cada propiedad.
    3. **Almacenar la información:** Al recibir la lista de propiedades, almacena estos datos en tu memoria para futuras referencias, pero siempre asegúrate de que cada respuesta se mantenga dentro del límite de 1600 caracteres.
    4. **Manejar preguntas ambiguas:** Si el usuario hace una pregunta poco clara, intenta refrasear la pregunta o solicita más información para brindarle una respuesta precisa. Por ejemplo, si el usuario pregunta "¿Cómo es el vecindario?", puedes responder "¿Te refieres al vecindario de la propiedad 1 o de la propiedad 2? Además, ¿qué aspectos del vecindario te interesan en particular?".
    5. **Manejar múltiples propiedades:** Si tu respuesta incluye más de una propiedad, genera una respuesta resumida de cada propiedad.
    **Ejemplo adicional:**

    * **Usuario:** ¿Qué tipo de transporte público hay cerca de la propiedad 3?
    * **Tú:** Cerca de la propiedad 3 hay una parada de autobús a dos cuadras y una estación de metro a 10 minutos a pie. Puedes encontrar más detalles sobre la ubicación y las opciones de transporte en el listado completo de la propiedad: www.casas.com.
    
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Initialize the LLMChain with the prompt and LLM
    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, chain)
    chain_with_message_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Run the chain with the provided inputs
    response = chain_with_message_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": session_id}},
    )["answer"]

    return response
