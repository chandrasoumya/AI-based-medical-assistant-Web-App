from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from googlesearch import search
import re
from itertools import combinations
from collections import Counter
import operator
import os
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

# Load OpenRouter API Key from environment variable
API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# Load disease prediction model and datasets
try:
    with open('data/disease_model.pkl', 'rb') as file:
        lr = pickle.load(file)
    df_norm = pd.read_csv('data/dis_sym_dataset_norm.csv')
    dataset_symptoms = df_norm.columns[1:].tolist()
    df_comb = pd.read_csv('data/dis_sym_dataset_comb.csv')
except FileNotFoundError as e:
    print(f"Error: One or more required files not found! Details: {str(e)}")
    sys.exit(1)
except pickle.UnpicklingError as e:
    print(f"Error: Failed to unpickle 'disease_model.pkl'. Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading model/datasets: {str(e)}")
    sys.exit(1)

# Load chest X-ray model
try:
    xray_model = load_model('data/chest_xray.h5')
except FileNotFoundError as e:
    print(f"Error: Chest X-ray model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading chest X-ray model: {str(e)}")
    sys.exit(1)

# Load COVID-19 X-ray model
try:
    covid_model = load_model('data/covid_final.h5')
except FileNotFoundError as e:
    print(f"Error: COVID-19 model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading COVID-19 model: {str(e)}")
    sys.exit(1)

# Load skin disease model
try:
    skin_model = load_model('data/trial_skin.h5')
except FileNotFoundError as e:
    print(f"Error: Skin disease model file not found! Details: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading skin disease model: {str(e)}")
    sys.exit(1)

# Class labels for X-ray diseases
class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Class labels for COVID-19 X-ray
covid_classes = ["COVID-19", "Pneumonia", "Normal"]

# Full names of skin disease classes
skin_classes = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Skin Lesion"
}
skin_class_labels = list(skin_classes.keys())
SKIN_IMG_SIZE = (224, 224)  # From reference code

# Utilities for symptom processing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Preprocessing function for images
def preprocess_image(image_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function for general X-ray
def predict_xray(image_path, model, class_labels):
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = predictions[0][top_3_indices] * 100
        top_3_classes = [class_labels[i] for i in top_3_indices]
        return [{"disease": cls, "confidence": float(round(conf, 1))} for cls, conf in zip(top_3_classes, top_3_confidences)]
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

# Prediction function for COVID-19 X-ray
def predict_covid_xray(image_path, model, class_labels, top_k=3):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)[0]
    temperature = 0.5
    scaled_preds = np.exp(predictions / temperature) / np.sum(np.exp(predictions / temperature))
    top_indices = np.argsort(scaled_preds)[-top_k:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_probs = [float(round(scaled_preds[i] * 100, 2)) for i in top_indices]
    return [{"disease": label, "confidence": prob} for label, prob in zip(top_labels, top_probs)]

# Prediction function for skin disease
def predict_skin(image_path, model, class_labels):
    try:
        img_array = preprocess_image(image_path, target_size=SKIN_IMG_SIZE)
        predictions = model.predict(img_array)
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = predictions[0][top_3_indices] * 100
        top_3_classes = [class_labels[i] for i in top_3_indices]
        return [{"disease": skin_classes[cls], "confidence": float(round(conf, 1))} for cls, conf in zip(top_3_classes, top_3_confidences)]
    except Exception as e:
        raise Exception(f"Skin prediction failed: {str(e)}")

# Chatbot query function with formatting
def query_gemini(prompt, image_url=None):
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "your_site_url",
            "X-Title": "your_site_name"
        }
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if image_url:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
        payload = {"model": "google/gemini-2.0-pro-exp-02-05:free", "messages": messages}
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            raw_response = response_data["choices"][0]["message"]["content"]
            return format_response(raw_response)
        else:
            return "Error: No valid response received from Gemini Pro."
    except Exception as e:
        return f"Error: {str(e)}"

def format_response(response_text):
    response_text = response_text.replace("\n\n", "\n")
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response_text)
    response_text = re.sub(r"\*(.*?)\*", r"<strong>\1</strong>", response_text)
    response_text = re.sub(r"\n\* (.*?)", r"\n• \1", response_text)
    response_text = response_text.replace("\n", "<br>")
    return response_text

# Symptom processing functions
def synonyms(term):
    synonyms = []
    try:
        response = requests.get(f'https://www.thesaurus.com/browse/{term}', timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for li in soup.select('section.MainContentContainer div.css-191l5o0-ClassicContentCard li'):
            synonyms.append(li.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

def diseaseDetail(term):
    ret = {"disease": term, "details": {}}
    try:
        query = f"{term} site:wikipedia.org"
        search_results = list(search(query, num_results=1))
        if not search_results:
            return {"error": f"No Wikipedia page found for {term}"}
        url = search_results[0]
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        info_table = soup.find("table", {"class": "infobox"})
        if info_table:
            for row in info_table.find_all("tr"):
                th = row.find("th", {"scope": "row"})
                if th and th.text.strip() in ["Symptoms", "Causes", "Risk factors", "Diagnostic method", "Prevention", "Treatment", "Usual onset"]:
                    td = row.find("td")
                    if td:
                        value = re.sub(r'\[.*?\]', '', td.get_text(strip=True))
                        ret["details"][th.text.strip()] = value
        return ret
    except Exception as e:
        return {"error": f"Failed to fetch details: {str(e)}"}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/covid19')
def covid_xray():
    return render_template('covid19.html')

@app.route('/predict_xray', methods=['GET', 'POST'])
def predict_xray_route():
    if request.method == 'POST':
        if 'xray_image' not in request.files:
            return render_template('xray_result.html', error="No image uploaded")
        file = request.files['xray_image']
        if file.filename == '':
            return render_template('xray_result.html', error="No file selected")
        upload_folder = 'static/uploads'
        try:
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            predictions = predict_xray(file_path, xray_model, class_names)
            return render_template('xray_result.html', predictions=predictions, image_path=f'uploads/{file.filename}')
        except Exception as e:
            return render_template('xray_result.html', error=f"Analysis error: {str(e)}")
    return render_template('predict_xray.html')

@app.route('/predict_covid_xray', methods=['POST'])
def predict_covid_xray_route():
    if 'xray_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['xray_image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    predictions = predict_covid_xray(file_path, covid_model, covid_classes)
    return jsonify({"predictions": predictions, "image_path": f'uploads/{file.filename}'})

@app.route('/predict_skin', methods=['GET', 'POST'])
def predict_skin_route():
    if request.method == 'POST':
        if 'skin_image' not in request.files:
            return render_template('skin_result.html', error="No image uploaded")
        file = request.files['skin_image']
        if file.filename == '':
            return render_template('skin_result.html', error="No file selected")
        upload_folder = 'static/uploads'
        try:
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            predictions = predict_skin(file_path, skin_model, skin_class_labels)
            return render_template('skin_result.html', predictions=predictions, image_path=f'uploads/{file.filename}')
        except Exception as e:
            return render_template('skin_result.html', error=f"Analysis error: {str(e)}")
    return render_template('predict_skin.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get('user_input')
    image_url = data.get('image_url')
    response = query_gemini(user_input, image_url)
    return jsonify({'response': response})

@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    user_input = request.form.get('symptoms', '')
    user_symptoms = user_input.lower().split(',')
    processed_user_symptoms = [' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym.strip().replace('-', ' '))]) for sym in user_symptoms]
    expanded_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym) + 1):
            for subset in combinations(user_sym, comb):
                str_sym.update(synonyms(' '.join(subset)))
        str_sym.add(' '.join(user_sym))
        expanded_symptoms.append(' '.join(str_sym).replace('_', ' '))
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        data_sym_split = data_sym.split()
        for user_sym in expanded_symptoms:
            if sum(1 for symp in data_sym_split if symp in user_sym.split()) / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)
    return jsonify({"found_symptoms": list(found_symptoms)})

@app.route('/select_symptoms', methods=['POST'])
def select_symptoms():
    selected_indices = request.json.get('selected_indices', [])
    found_symptoms = request.json.get('found_symptoms', [])
    final_symp = [found_symptoms[int(idx)] for idx in selected_indices]
    dis_list = set()
    counter_list = []
    for symp in final_symp:
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))
    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
        row[0].pop(0)
        for idx, val in enumerate(row[0]):
            if val != 0 and dataset_symptoms[idx] not in final_symp:
                counter_list.append(dataset_symptoms[idx])
    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
    return jsonify({"selected_symptoms": final_symp, "related_symptoms": dict_symp_tup[:10]})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.json
    final_symptoms = data["final_symptoms"]
    sample_x = [0] * len(dataset_symptoms)
    for symp in final_symptoms:
        if symp in dataset_symptoms:
            sample_x[dataset_symptoms.index(symp)] = 1
    sample_x = np.array(sample_x).reshape(1, -1)
    prediction = lr.predict_proba(sample_x)
    diseases = list(set(df_norm['label_dis']))
    diseases.sort()
    top5 = np.argsort(prediction[0])[::-1][:5]
    result = [{"disease": diseases[idx], "probability": round(prediction[0][idx] * 120, 1)} for idx in top5]
    return jsonify({"predictions": result})

@app.route('/disease_details', methods=['POST'])
def disease_details():
    disease = request.form.get('disease', '')
    if not disease:
        return jsonify({"error": "No disease provided"}), 400
    details = diseaseDetail(disease)
    print(f"Fetching details for disease: {disease}")
    if "details" not in details:
        print(f"Error fetching details: {details}")
        return jsonify({"error": "Failed to retrieve disease details"}), 500
    return jsonify({"disease": disease, "details": details["details"]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)