from flask import Blueprint, request
from twilio.rest import Client
import sys
import os
from .utils import load_properties_with_embeddings, search_properties_with_embeddings, classify_intent
from .langchain_ import query_llm

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Initialize Twilio client with credentials from environment variables
client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

# Create a Blueprint for the routes
bp = Blueprint('routes', __name__)

@bp.route('/', methods=['GET'])
def home():
    return "Welcome to the WhatsApp bot server!"


@bp.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').lower()
    from_number = request.values.get('From', '')  # Get the sender's WhatsApp number
    print(f"----Incoming message: {incoming_msg} from {from_number}----")  # Debugging line

    # Classify intent of incoming message
    intent = classify_intent(incoming_msg)
    print(f"----INTENT:{intent}----")

    # Prepare the response based on the classified intent
    if intent == 'contact agent':
        response_message = "En unos minutos, el agente inmobiliario se pondrá en contacto contigo. Gracias por contactar a Asesores Inmobiliarios."
        # Send the response using Twilio's REST API
        try:
            sent_message = client.messages.create(
                from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
                body=response_message,
                to=from_number
            )
            print(f"Message sent with SID: {sent_message.sid}")  # Debugging line
        except Exception as e:
            print(f"Failed to send message: {e}")  # Debugging line

        return sent_message.body
    else: 
        # Load properties with embeddings
        properties_df = load_properties_with_embeddings()

        # Search for relevant properties using embeddings
        relevant_properties_df = search_properties_with_embeddings(properties_df, incoming_msg)

        # Convert relevant properties to a string for the LLM
        relevant_properties_str = relevant_properties_df['property_string'].to_json()

        #Query the LLM
        llm_response = query_llm(properties=relevant_properties_str, user_message=incoming_msg)
        print(f"LLM response: {llm_response}")

        # Send the response using Twilio's REST API
        try:
            sent_message = client.messages.create(
                from_='whatsapp:+14155238886',  # Your Twilio WhatsApp number
                body=llm_response,
                to=from_number
            )
            print(f"Message sent with SID: {sent_message.sid}")  # Debugging line
        except Exception as e:
            print(f"Failed to send message: {e}")  # Debugging line

        return sent_message.body 

# if __name__ == "__main__":
# #     #test chat history
#     test_query = "Tiene propiedades en la urbanizacion san rafael?"
#     properties_df = load_properties_with_embeddings()
#     print(search_properties_with_embeddings(properties_df, test_query))