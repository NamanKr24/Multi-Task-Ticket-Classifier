import joblib, json, re, string
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gradio as gr
from scipy.sparse import hstack

issue_clf = joblib.load("issue_model.pkl")
urgency_clf = joblib.load("urgency_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

with open("product_list.json") as f:
    product_list = json.load(f)

with open("complaint_keywords.json") as f:
    complaint_keywords = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b'

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_entities(text):
    text_lower = text.lower()
    entities = {
        "product": None,
        "dates": re.findall(date_pattern, text),
        "complaint_keywords": [kw for kw in complaint_keywords if kw in text_lower]
    }
    for product in product_list:
        if product.lower() in text_lower:
            entities["product"] = product
            break
    return entities

def process_ticket(ticket_text):
    cleaned = preprocess(ticket_text)
    vec = tfidf.transform([cleaned])
    extra = np.array([[len(ticket_text), TextBlob(ticket_text).sentiment.polarity]])
    full = hstack([vec, extra])
    issue = issue_clf.predict(full)[0]
    urgency = urgency_clf.predict(full)[0]
    return {
        "Predicted Issue Type": issue,
        "Predicted Urgency Level": urgency,
        "Extracted Entities": extract_entities(ticket_text)
    }

iface = gr.Interface(
    fn=process_ticket,
    inputs=gr.Textbox(lines=5, placeholder="Enter support ticket text here..."),
    outputs="json",
    title="Customer Ticket Classifier"
)

iface.launch()
