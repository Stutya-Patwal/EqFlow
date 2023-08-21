# modules
import requests
from bs4 import BeautifulSoup

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

#nltk.download('punkt')
#nltk.download('stopwords')

def cleantext(sample_text):

    words = word_tokenize(sample_text)

    # Remove stopwords and non-alphanumeric words
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]

    # Join the cleaned words to form human-readable text
    cleaned_text = ' '.join(cleaned_words)

    # Sentiment analysis using TextBlob
    sentiment_score = TextBlob(cleaned_text).sentiment.polarity

    # Create a DataFrame for analysis
    data = {'Text': [cleaned_text], 'Sentiment': [sentiment_score]}
    df = pd.DataFrame(data)

    # Vectorize the text using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Text'])
    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


    return cleaned_text, sentiment_score



def _get_response(url, headers=None, timeout=None):
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response

    except requests.exceptions.RequestException as e:
        print("Error occurred while making the request to the URL:", e)
        return None

def verify_url(url):
    response = _get_response(url)
    if response:
        status_code = response.status_code

        if status_code == 200:
            print("URL is accessible and the request was successful (200 OK)")
        elif status_code == 404:
            print("URL could not be found on the server (404 Not Found)")
        elif status_code == 403:
            print("The server understood the request, but refused to authorize it (403 Forbidden)")
        elif status_code == 500:
            print("The server encountered an error while processing the request (500 Internal Server Error)")
        else:
            print("Unknown Error: Check the URL or server response")

def get_soup(url, headers=None, timeout=None):
    response = _get_response(url, headers=headers, timeout=timeout)
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    return None

def get_raw_text(soup):
    if soup:
        all_text = soup.get_text()
        return all_text
    return ""

def get_processed_text(soup):
    if soup:
        all_text = soup.get_text()
        cleaned_text = clean(all_text)
        return cleaned_text
    return ""

def find(url, tag=None, id_=None, class_=None, text=None, headers=None, timeout=None):
    response = _get_response(url, headers=headers, timeout=timeout)
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        filters = {}

        if tag:
            filters['name'] = tag
        if id_:
            filters['id'] = id_
        if class_:
            filters['class'] = class_
        if text:
            filters['text'] = text

        elements = soup.find_all(**filters)
        return elements
    return []

def clean(text):
    cleaned_text = clean(text)
    return cleaned_text