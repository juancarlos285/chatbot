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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Ensure the OpenAI API key is set as an environment variable
openai_api_key = config.OPENAI_API_KEY

# Initialize OpenAI LLM with LangChain
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o-mini")
print('Working LLM' if llm else 'no LLM')

# Session history storage
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def query_llm(properties, input, session_id):
    # Turns properties into runnable list of Documents
    runnable_listings = RunnableLambda(lambda *args: [Document(page_content=listing, metadata={'source': 'local'}) for listing in properties])
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
            (
                "system",
                template
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human", "{input}" 
            )
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
    output_messages_key="answer"
    )

    # Run the chain with the provided inputs
    response = chain_with_message_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": session_id}},
    )["answer"]
    
    return response

listing_1 = '''
        location: Las Cucardas,
        neighborhood: El Inca, Quito,
        area: 145 m² tot,
        price: USD 450,
        fee: USD 1 Alícuota,
        bedrooms: 2 estac.,
        bathroom: N/A,
        parking_spots: N/A,
        description: Arriendo Local comercial - bodega de 145m2 en el Sector de El Inca, es apta para taller de confección textil, serigrafía, taller mecánico, taller de motos, área de logística, transporte, transporte de encomiendas, archivo, almacenamiento de materiales eléctricos, madera, muebles, insumos textiles, repuestos, llantas, productos no perecibles, etc. Excelente ubicación con salida hacia la Av. El Inca, Av. 6 de Diciembre. Descripción: El local - bodega tiene 145 m2 ubicado en primer piso con un baño y dos estacionamientos, área para carga y descarga de mercadería, seguridad en su ingreso. Medidor independiente de luz. precio: $450 y 2 meses de Garantía. citas: Ver datos cbr Eveling Rosas 0 9 8 7 0 0 6 8 0 6. #bodegadearriendo, #rentobodegaelinca, #galpon, #bodegaarchivo, #bodegaalmacenamiento, #localparabodegadearriendo, #wescobodegaelinca, #arriendolocalparatallertextil, #bodegagym, #arriendoparaencomiendas, #arriendoparatransporte, #tallerdemotos, #motosreparacion, #cbrevelingrosas,
        url: plusvalia.com/propiedades/clasificado/alclbgin-local-comercial-bodega-145-m-sup2--sector-el-inca-90152077.html,
        id: 1
    '''
listing_2 = '''
    location: Gonzalez Súarez,
    neighborhood: "González Suárez, Quito,
    area: "344 m² tot.,
    price: USD 2.262,
    fee: "USD 1 Alícuota,
    bedrooms: 3 hab.,
    bathrooms: 4 baños,
    parking_spots: 4 estac.,
    description: Valor renta $2262, 00 (incluye alícuota) Departamento de lujo en piso alto (P6), espectacular vista de la ciudad, ubicación estratégica, 3 dormitorios. A pocos metros del puente hacia Bellavista y del túnel Guayasamín que comunica la ciudad con los valles de Cumbayá, Tumbaco, Puembo y la vía al Aeropuerto en minutos. El entorno se caracteriza por ser altamente residencial sin embargo en la cercanía se cuenta con todos los servicios comerciales, gastronómicos, bancarios, zonas culturales y de recreación. áreas: 250, 40m2 habitables. 40, 50m2 balcón que rodea el dpto. 50 m2 Cuatro (4) parqueaderos cubiertos. 4 m2 Una (1) bodega. características: Ambientes iluminados con ventanales piso-techo. Amplias áreas sociales, sala – comedor con salida a balcón. Moderna cocina equipada con plancha a inducción, extractor, refrigeradora 2 puertas, microondas, horno (empotrados), funcionales muebles altos y bajos, isla desayunador. Area de lavado independiente. Cuarto y baño de servicio. Baño social. Sala de estar, familiar o de tv, con amplio espacio para adecuar área de trabajo o estudio. Dormitorio máster con gran vestidor iluminado, baño privado que incluye espectacular tina y salida a balcón. 2 dormitorios con baño privado c/u y espaciosos closets y/o vestidores. acabados de lujo: Pisos de madera lacada en zonas sociales y de descanso. Porcelanato importado en baños, cocina y balcón. Sanitarios, lavabos y grifería gama alta. Mesones de cocina y baños en Quarzo Silestone. Ventanería de vidrio flotado de 6mm de espesor. El edificio cuenta con: 2 ascensores. 3 subsuelos de estacionamientos. Guardianía 24/7. Cámaras de seguridad en todas las áreas. Gimnasio. Sala de cine con cómodas butacas. Kids club. Sala comunal. Terraza con moderna área de bbq cubierta. Estructura sismo resistente. Sistema contra incendios. Agua caliente centralizada. Sistema automatizado de ingreso peatonal y vehicular. Transformador y generador eléctrico. citas: Ver datos cbr Eveling Rosas Lic. Profesional 1214-P. #rentagonzalezsuarez, #dptoexclusivo, #dptoaestrenargonzalezsuarez, #arriendodepartamentogonzalezsuarez, #arriendodptotunelguayasamin, #dptodelujo, #dptohotelquito, #serentaexclusivodepartamentotresdormitorios, #arriendodptoconvistagonzalezsuarez, #dptoarriendogonzalez, #inversiongonzalezsuarez, #diplomaticos, #ong, #embajadausa, #embajadasyconsulados, #exclusividad, #ampliodepartamento, #departamentodelujo, #departamentoconvista, #realtor, #inversion, #arriendodepartamentodelujo, #arriendogonzalezsuarez, #arriendodptoexclusivo, #dptoarriendolujo,
    url: "plusvalia.com/propiedades/clasificado/alclapin-estrenar-dpto-de-lujo-en-renta-en-edificio-moderno-143153358.html",
    id: 3
    '''

# user_message = "precio de la propiedad por el inca por favor"
# llm_response = query_llm(runnable_listings, user_message, 'abc123')
# print(llm_response)

# user_message_2 = "cuántos metros cuadrados tiene?"
# second_response = query_llm(runnable_listings, user_message_2, 'abc123')
# print(second_response)

# user_message_3 = "y tiene propiedad en los chillos?"
# third_response = query_llm(runnable_listings, user_message_3, 'abc123')
# print(third_response)

# user_message_4 = "me das información del apartamento en la gonzalez suarez"
# fourth_response = query_llm(runnable_listings, user_message_4, 'abc123')
# print(fourth_response)

# user_message_5 = "me repites el precio de la propiedad con id 1"
# fifth_response = query_llm(runnable_listings, user_message_5, 'abc123')
# print(fifth_response)

