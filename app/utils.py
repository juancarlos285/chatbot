import re
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_openai import OpenAIEmbeddings
from twilio.rest import Client
import nltk
from nltk.corpus import stopwords

# Set of stopwords in Spanish
nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def property_data():
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "property_listings.json")
    )
    with open(file_path, "r") as file:
        properties = json.load(file)
    return properties


def load_properties_with_embeddings():
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "openai_embeddings.csv")
    )
    df = pd.read_csv(file_path, converters={"embedding": eval})
    return df


# Function that will clean the data to be embedded
def clean_and_transform_data(data):
    # Create a new dictionary to store cleaned data
    cleaned_data = {}

    # Copy all keys except "page" and "url"
    for key, value in data.items():
        if key not in ["page", "url"]:
            cleaned_data[key] = value

    # Normalize, and remove stop words
    cleaned_data["description"] = clean_text(data["description"])

    # Extract and clean numerical values (bedrooms, bathrooms, parking spots)
    cleaned_data["bedrooms"] = (
        int(re.findall(r"\d+", data["bedrooms"])[0])
        if data["bedrooms"] != "N/A"
        else None
    )
    cleaned_data["bathrooms"] = (
        int(re.findall(r"\d+", data["bathrooms"])[0])
        if data["bathrooms"] != "N/A"
        else None
    )
    cleaned_data["parking_spots"] = (
        int(re.findall(r"\d+", data["parking_spots"])[0])
        if data["parking_spots"] != "N/A"
        else None
    )

    # Clean location and neighborhood (directly using clean_text)
    cleaned_data["location"] = clean_text(data["location"])
    cleaned_data["neighborhood"] = clean_text(data["neighborhood"])

    cleaned_data_to_str = f"""
        ubicación: {cleaned_data['location']}
        barrio: {cleaned_data['neighborhood']}
        area: {cleaned_data['area']}
        precio: {cleaned_data['price']}
        alícuota: {cleaned_data['fee']}
        habitaciones: {cleaned_data['bedrooms']}
        baños: {cleaned_data['bathrooms']}
        parqueaderos: {cleaned_data['parking_spots']}
        {cleaned_data['description']}
        """

    return cleaned_data_to_str


def clean_text(text):
    text = text.lower()  # Normalize to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    # Remove stop words
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


# Function that will format the property to be used by the LLM
def property_to_string(property):
    return f"""ID:{property['id']}
            Ubicación: {property['location']}
            Barrio: {property['neighborhood']} 
            Area: {property['area']}
            Precio: {property['price']}
            Alícuota: {property['fee']}
            Habitaciones: {property['bedrooms']}
            Baños: {property['bathrooms']}
            Parqueaderos: {property['parking_spots']}
            Descripción: {property['description']}
            Enlace: {property['url']}
            """


def openai_embeddings(texts):
    OPENAI_API_KEY = None
    embed_model = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=OPENAI_API_KEY
    )
    return embed_model.embed_documents(texts)


def save_properties_to_csv(properties, file_name):
    property_strings = [property_to_string(property) for property in properties]

    property_embeddings = [
        clean_and_transform_data(property) for property in properties
    ]
    property_embeddings = openai_embeddings(property_embeddings)

    df = pd.DataFrame(
        {"property_string": property_strings, "embedding": property_embeddings}
    )

    output_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", file_name)
    )
    df.to_csv(output_file_path, index=False)


def search_properties_with_embeddings(properties_df, query):
    query_embedding = np.array(openai_embeddings([query])).reshape(1, -1)
    property_embeddings = np.stack(properties_df["embedding"].values)

    similarities = cosine_similarity(query_embedding, property_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # Get top 5 most similar properties
    top_properties = properties_df.iloc[top_indices]["property_string"].values

    return top_properties


# ----Load properties to generate embeddings and save to CSV----
# properties = property_data()
# save_properties_to_csv(properties, 'properties_with_embeddings.csv')


# Load agent information
def load_agent_data():
    with open("data/agent_data.json", "r") as f:
        return json.load(f)


# Get agent information based on property ID
def get_agent_info(property_id):
    agents = load_agent_data()
    return agents[property_id]


# Get property info for the agent based on the id provided by the client
def get_property_for_agent(id):
    properties = pd.read_json("data/property_listings.json")
    retrieved_property = properties[properties["id"] == int(id)]
    location = retrieved_property["location"].values[0]
    neighborhood = retrieved_property["neighborhood"].values[0]
    return (location, neighborhood)


# Send a message to the agent with the customer's details
def send_message_to_agent(agent_phone_number, message):
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    try:
        sent_message = client.messages.create(
            from_=config.TWILIO_SANDBOX_NUMBER,  # Your Twilio WhatsApp number
            body=message,
            to=f"whatsapp:{agent_phone_number}",
        )
        print(f"Message sent to agent with SID: {sent_message.sid}")
    except Exception as e:
        print(f"Failed to send message to agent: {e}")


# ----Intent Classification----
model_path = "./results/model-6"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model.eval()


def classify_intent(text, model, tokenizer):
    text_cleaned = clean_text(text)
    inputs = tokenizer(text_cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return "contact agent" if predicted_class_id == 0 else "other"
