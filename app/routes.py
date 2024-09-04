from flask import Blueprint, request
from twilio.rest import Client
import sys
import os
import time
import pandas as pd
from .utils import load_properties_with_embeddings, search_properties_with_embeddings, classify_intent, send_message_to_agent, get_agent_info, get_property_for_agent
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

#Context dictionary to store user states
user_context = {}

@bp.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').lower()
    from_number = request.values.get('From', '')  # Get the sender's WhatsApp number
    print(f"----Incoming message: {incoming_msg} from {from_number}----")

    # Check if the user is in 'expecting property id' context
    if user_context.get(from_number) == 'awaiting_property_id':
        # Process the property ID
        try:
            property_id = incoming_msg.strip()
            location, neighborhood = get_property_for_agent(property_id)
            agent_info = get_agent_info(property_id)
            if agent_info:
                # Notify the agent with customer details
                agent_message = (f"Un cliente con el número {from_number} ha solicitado información sobre la propiedad en {location}\n{neighborhood}. "
                                 "Por favor, contacta al cliente para más detalles.")
                send_message_to_agent(agent_info['phone_number'], agent_message)
                time.sleep(5)
                response_message = "Gracias. Hemos notificado al agente. Se pondrán en contacto contigo pronto."
            else:
                response_message = "Lo siento, no pude encontrar un agente asociado a ese ID de propiedad. Por favor verifica el ID."
        except ValueError:
            response_message = "Parece que el ID de propiedad proporcionado no es válido. Por favor intenta de nuevo con un número válido."

        # Reset the user's context
        user_context[from_number] = None

        # Send the response using Twilio's REST API
        try:
            sent_message = client.messages.create(
                from_= config.TWILIO_SANDBOX_NUMBER,  # Your Twilio WhatsApp number
                body=response_message,
                to=from_number
            )
            print(f"Message sent with SID: {sent_message.sid}")
        except Exception as e:
            print(f"Failed to send message: {e}") 

        return sent_message.body
    
    else:
        # Classify intent of incoming message
        intent = classify_intent(incoming_msg)
        print(f"----INTENT:{intent}----")

        # Prepare the response based on the classified intent
        if intent == 'contact agent':
            response_message = ("En unos minutos, el agente inmobiliario se pondrá en contacto contigo. "
                                "Por favor, envíanos el número de ID de la propiedad que deseas visitar para confirmar la solicitud.")
            # Set context to awaiting_property_id for this user
            user_context[from_number] = 'awaiting_property_id'

            # Send the response using Twilio's REST API
            try:
                sent_message = client.messages.create(
                    from_= config.TWILIO_SANDBOX_NUMBER,  # Your Twilio WhatsApp number
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
            relevant_properties_str = "\n\n".join(relevant_properties_df['property_string'].values)

            #Query the LLM
            llm_response = query_llm(properties=relevant_properties_str, user_message=incoming_msg)
            print(f"LLM response: {llm_response}")

            # Send the response using Twilio's REST API
            try:
                sent_message = client.messages.create(
                    from_= config.TWILIO_SANDBOX_NUMBER,  # Your Twilio WhatsApp number
                    body=llm_response,
                    to=from_number
                )
                print(f"Message sent with SID: {sent_message.sid}")  # Debugging line
            except Exception as e:
                print(f"Failed to send message: {e}")  # Debugging line

            return sent_message.body 
