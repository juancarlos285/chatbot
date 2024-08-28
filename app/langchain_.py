from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import sys

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Ensure the OpenAI API key is set as an environment variable
openai_api_key = config.OPENAI_API_KEY

# Initialize OpenAI LLM with LangChain
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")
print('Working LLM' if llm else 'no LLM')

chat_history_for_chain = ChatMessageHistory()

def query_llm(properties, user_message):
    # Define the prompt template
    template = """
    Eres un asistente de bienes raíces experto en las propiedades: {properties} 

    Si la pregunta no es acerca de inmobiliaria, recuerda tu rol al usuario y no respondas la pregunta.

    Tus tareas son:

    1. **Responder preguntas concisas:** Responde a las preguntas en **1600 caracteres o menos**. **Es crucial** que la respuesta **no** exceda este límite. Si es necesario, recorta la información para ajustarte a este límite.
    2. **SIEMPRE proporciona el ID de cada propiedad.
    3. **Almacenar la información:** Al recibir la lista de propiedades, almacena estos datos en tu memoria para futuras referencias, pero siempre asegúrate de que cada respuesta se mantenga dentro del límite de 1600 caracteres.
    4. **Manejar preguntas ambiguas:** Si el usuario hace una pregunta poco clara, intenta refrasear la pregunta o solicita más información para brindarle una respuesta precisa. Por ejemplo, si el usuario pregunta "¿Cómo es el vecindario?", puedes responder "¿Te refieres al vecindario de la propiedad 1 o de la propiedad 2? Además, ¿qué aspectos del vecindario te interesan en particular?".

    **Ejemplo adicional:**

    * **Usuario:** ¿Qué tipo de transporte público hay cerca de la propiedad 3?
    * **Tú:** Cerca de la propiedad 3 hay una parada de autobús a dos cuadras y una estación de metro a 10 minutos a pie. Puedes encontrar más detalles sobre la ubicación y las opciones de transporte en el listado completo de la propiedad: www.casas.com.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                template
            ),
            (
                "placeholder", "{chat_history}"
            ),
            (
                "human", "{user_message}" 
            )
        ]
    )

    # Initialize the LLMChain with the prompt and LLM
    chain = prompt | llm

    print("-----CHAT HISTORY-----")
    print(chat_history_for_chain.messages)

    chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history_for_chain,
    input_messages_key="user_message",
    history_messages_key="chat_history",
)

    # Run the chain with the provided inputs
    response = chain_with_message_history.invoke(
        {"user_message": user_message,
         "properties": properties,
         "chat_history": chat_history_for_chain
         },
        {"configurable": {"session_id": "unused"}},
    )
    
    chat_history_for_chain.add_user_message(user_message)
    chat_history_for_chain.add_ai_message(response.content)

    return response.content

