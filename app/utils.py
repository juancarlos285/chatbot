import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import re
import random
import time
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.embeddings import OpenAIEmbeddings
from twilio.rest import Client
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 11; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
        "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A5341f Safari/604.1",
        "Mozilla/5.0 (Linux; Android 10; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.93 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 13_3_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.5 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.2 Safari/605.1.15"
    ]
    return random.choice(user_agents)

def save_data_to_json(data, filename):
    # Construct the path to the data folder at the same level as the app folder
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(project_root, 'data')
    os.makedirs(data_folder, exist_ok=True)
    filepath = os.path.join(data_folder, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filepath}")



def get_property_listings():
    base_url = "https://www.plusvalia.com/inmobiliarias/asesores-inmobiliarios_51288661-inmuebles"
    
    properties = []
    previous_listings = set()
    scraped_urls = set()
    page = 1

    session = requests.Session()
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    session.headers.update(headers)

    while True:
        if page == 1:
            url = f"{base_url}.html"
        else:
            url = f"{base_url}-pagina-{page}.html"

        #ScraperAPI URL
        scraperapi_url = f"http://api.scraperapi.com?api_key={config.SCRAPERAPI_KEY}&url={url}&country_code=us"
        
        try:
            response = session.get(url)
            if response.status_code == 403:
                print(f"403 Forbidden error on page {page}. Retrying after delay.")
                time.sleep(random.uniform(5, 10))  # Random delay between 5 and 10 seconds
                continue  # Retry the same page
            response.raise_for_status()

            if url in scraped_urls:
                print(f"URL {url} has already been scraped. Skipping.")
                page += 1
                scraped_urls.add(url)
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the main container that holds all the postings
            postings_container = soup.find('div', class_='postings-container')
            if not postings_container:
                print(f"No postings container found on page {page}. Stopping the scraper.")
                break
            
            # Find all individual listings within the main container
            listings = postings_container.find_all('div', class_='CardContainer-sc-1tt2vbg-5 fvuHxG')

            if not listings:
                print(f"No listings found on page {page}. Stopping the scraper.")
                break

            # Check if the listings are the same as the previous page (to detect the end)
            current_listings = set(listing.get_text() for listing in listings)
            if current_listings == previous_listings:
                print(f"Same listings found on page {page}. Assuming end of listings.")
                break

            # Update the previous listings
            previous_listings = current_listings

            # Debugging: Print the number of listings found
            print(f"Found {len(listings)} listings on page {page}")

            for listing in listings:
                # Extract location
                location_div = listing.find('div', class_=re.compile(r'LocationAddress.*postingAddress'))
                location = location_div.text.strip() if location_div else "N/A"
               
                # Extract neighborhood
                neighborhood_h2 = listing.find('h2', class_=re.compile(r'LocationLocation-sc-ge2uzh-2 fziprF'))
                neighborhood = neighborhood_h2.text.strip() if neighborhood_h2 else "N/A"
                
                # Extract main features (area, bedrooms, bathrooms, parking spots)
                features_h3 = listing.find('h3', class_='PostingMainFeaturesBlock-sc-1uhtbxc-0 cHDgeO')
                if features_h3:
                    spans = features_h3.find_all('span')
                    area = spans[0].text.strip() if len(spans) > 0 else "N/A"
                    bedrooms = spans[1].text.strip() if len(spans) > 1 else "N/A"
                    bathrooms = spans[2].text.strip() if len(spans) > 2 else "N/A"
                    parking_spots = spans[3].text.strip() if len(spans) > 3 else "N/A"
                else:
                    area = bedrooms = bathrooms = parking_spots = "N/A"
                
                # Extract price
                price_div = listing.find('div', class_=re.compile(r'Price-sc-12dh9kl-3 geYYII'))
                price = price_div.text.strip() if price_div else "N/A"

                # Extract fee
                fee_div = listing.find('div', class_=re.compile(r'Expenses-sc-12dh9kl-1 iboaIF'))
                fee = fee_div.text.strip() if fee_div else "N/A"
                
                # Extract description
                description_h3 = listing.find('h3', class_='PostingDescription-sc-i1odl-11 fECErU')
                description = description_h3.find('a').text.strip() if description_h3 else "N/A"

                #Extract property link
                property_link = description_h3.find('a')['href'] if description_h3 else "N/A"
                
                # Store the details in a dictionary
                property_details = {
                    "page": page,
                    "location": location,
                    "neighborhood": neighborhood,
                    "area": area,
                    "price": price,
                    "fee": fee,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "parking_spots": parking_spots,
                    "description": description,
                    "url": "plusvalia.com" + property_link
                }
                properties.append(property_details)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching property listings from {url}: {e}")
            break

        # Add a random delay between requests to avoid rate limiting
        time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds
        page += 1

    return properties

#----Use when retrieving properties from plusvalia.com-----
# properties = get_property_listings()
# save_data_to_json(properties, 'property_listings.json')

def property_data():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'property_listings.json'))
    with open(file_path, 'r') as file:
        properties = json.load(file)
    return properties

def load_properties_with_embeddings():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'openai_embeddings.csv'))
    df = pd.read_csv(file_path, converters={'embedding': eval})
    return df

def clean_and_transform_data(data):
  """
  This function cleans and transforms a real estate data dictionary.

  Args:
      data (dict): The real estate data dictionary.

  Returns:
      dict: The updated dictionary with cleaned and transformed data.
  """
 # Create a new dictionary to store cleaned data
  cleaned_data = {}

  # Copy all keys except "page" and "url"
  for key, value in data.items():
    if key not in ["page", "url"]:
      cleaned_data[key] = value

  # Clean description: normalize, and remove stop words
  cleaned_data['description'] = clean_text(data['description'])

  # Extract and clean numerical values (bedrooms, bathrooms, parking spots)
  cleaned_data["bedrooms"] = int(re.findall(r"\d+", data["bedrooms"])[0]) if data["bedrooms"] != 'N/A' else None
  cleaned_data["bathrooms"] = int(re.findall(r"\d+", data["bathrooms"])[0]) if data['bathrooms'] != 'N/A' else None
  cleaned_data["parking_spots"] = int(re.findall(r"\d+", data["parking_spots"])[0]) if data["parking_spots"] != 'N/A' else None  

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
  text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
  # Remove stop words
  text = " ".join([word for word in text.split() if word not in stop_words])
  return text

def property_to_string(property):
    return f"""Ubicación: {property['location']}
            Barrio: {property['neighborhood']} 
            Area: {property['area']}
            Precio: {property['price']}
            Alícuota: {property['fee']}
            Habitaciones: {property['bedrooms']}
            Baños: {property['bathrooms']}
            Parqueaderos: {property['parking_spots']}
            Descripción: {property['description']}
            Enlace: {property['url']}
            ID: {property['id']}
            """

def get_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(texts)

def openai_embeddings(texts):
    OPENAI_API_KEY = None
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key= OPENAI_API_KEY)
    return embed_model.embed_documents(texts)

def save_properties_to_csv(properties, file_name):
    property_strings = [property_to_string(property) for property in properties]
    
    property_embeddings = [clean_and_transform_data(property) for property in properties]
    property_embeddings = openai_embeddings(property_embeddings)
    
    # Convert embeddings to list of lists
    embeddings_list = property_embeddings

    # Create DataFrame
    df = pd.DataFrame({
        'property_string': property_strings,
        'embedding': embeddings_list
    })

    # Save to CSV
    output_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', file_name))
    df.to_csv(output_file_path, index=False)

def search_properties_with_embeddings(properties_df, query):
    query_embedding = np.array(openai_embeddings([query])).reshape(1, -1)
    property_embeddings = np.stack(properties_df['embedding'].values)

    similarities = cosine_similarity(query_embedding, property_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]  # Get top 5 most similar properties

    return properties_df.iloc[top_indices]

#Load properties to generate embeddings and save to CSV
# properties = property_data()
# save_properties_to_csv(properties, 'properties_with_embeddings.csv')

# Load agent information
def load_agent_data():
    with open('data/agent_data.json', 'r') as f:
        return json.load(f)

# Get agent information based on property ID
def get_agent_info(property_id):
    agents = load_agent_data()
    return agents[property_id]

#Get property info for the agent based on the id provided by the client
def get_property_for_agent(id):
    properties = pd.read_json('data/property_listings.json')
    retrieved_property = properties[properties['id'] == int(id)]
    location = retrieved_property['location'].values[0]
    neighborhood = retrieved_property['neighborhood'].values[0]
    return (location, neighborhood)

# Send a message to the agent with the customer's details
def send_message_to_agent(agent_phone_number, message):
    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    try:
        sent_message = client.messages.create(
            from_= config.TWILIO_SANDBOX_NUMBER,  # Your Twilio WhatsApp number
            body=message,
            to=f'whatsapp:{agent_phone_number}'
        )
        print(f"Message sent to agent with SID: {sent_message.sid}") 
    except Exception as e:
        print(f"Failed to send message to agent: {e}")  

#----Intent Classification----
model_path = './results/model-6'  
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
def classify_intent(text):
    text_cleaned = clean_text(text)
    inputs = tokenizer(text_cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return 'contact agent' if predicted_class_id == 0 else 'other'
