#!/usr/bin/env python
# coding: utf-8

# In[15]:

import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Flask,jsonify, request,make_response
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flask_swagger_ui import get_swaggerui_blueprint
from flasgger import swag_from
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import pandas as pd
import re
import json
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.models import load_model
import tensorflow as tf
from nltk import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

# df = pd.read_csv("data/preproccesed.csv")

# In[ ]:

# model NN

# In[16]:
# class DeliveryEncoder(json.JSONEncoder):
# 	def default(self, obj):
		
# 		if isinstance(obj, Delivery):
# 			return { 'to' : obj.to,  'from' : obj.fr }

# 		return json.JSONEncoder.default(self, obj) # default, if not Delivery object. Caller's problem if this is not serialziable.

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False 
SWAGGER_URL = '/swagger'
API_URL = '/static/restapi.yml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "selamat datang regina"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

conn = sqlite3.connect('record.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS record(text varchar(255), text_clean varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS record_file(text varchar(255), text_clean varchar(255), sentiment varchar(255));''')
MODEL = joblib.load('mlpc.pkl')
MODELTF = load_model('tfmodel.h5')




# In[7]:


#welcomepage
@app.route('/', methods=['GET'])
def get():
  return "Presenting API for Sentiment Analysis!"



# In[17]:

# @swag_from("docs/hint.yml", methods=['POST'])
tfidf_vectorizer = joblib.load('feature.pkl') 
tfidf_vectorizerTF = joblib.load('featuretensorflow.pkl') 
vocab_size = 5000
max_length = 200
oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

@app.route('/textNN', methods=['POST'])
# NN processing
def text_sentimentNN():

    text = request.form.get('text').lower()
    clean_text = cleaning(text)
    # fitur = tfidf_vectorizer([clean_text])
    sent = MODEL.predict(tfidf_vectorizer.transform([clean_text]).toarray())
    # sent = str(MODEL.predict(fitur)[0])
    query = "INSERT INTO record (text, text_clean) VALUES (?,?)"
    val = (clean_text , str(sent))
    conn.execute(query , val)
    conn.commit()

           
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': clean_text, 
        'sentiment' : str(sent),
    }
    response_data = jsonify(json_response)
    return response_data

@app.route('/textTF', methods=['POST'])
# TF processing
def text_sentimentTF():

    text = request.form.get('text').lower()
    clean_text = cleaning(text)
    sent = MODELTF.predict(tfidf_vectorizerTF.transform([clean_text]).toarray())
    query = "INSERT INTO record (text, text_clean) VALUES (?,?)"
    val = (clean_text , str(sent))
    conn.execute(query , val)
    conn.commit() 

           
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': clean_text, 
        'sentiment' : str(sent),
    }
    response_data = jsonify(json_response)
    return response_data

@app.route('/fileNN', methods=['POST'])
def file_cleaningNN():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')
    
    # Cleaning file
    file_csvNN(datacsv)

    # file_cleaning(sent)

    # Define API response
    select_data = conn.execute("SELECT * FROM record_file")
    conn.commit
    data = [
        dict( text=row[0], text_clean=row[1] , sentiment = row[2])
    for row  in select_data.fetchall()
    ]
    
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': data, 
        
    }
    response_data = jsonify(json_response)
    return response_data

def file_csvNN(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert original text and cleaned text to sqlite database
        data_clean = cleaning(data_file)
        sent = MODEL.predict(tfidf_vectorizer.transform([data_clean]).toarray()).tolist()
        sent_dumb = json.dumps(sent)
        query = "insert into record_file (text,text_clean ,sentiment) values (?,?,?)"
        val = (data_file, data_clean ,sent_dumb )
        conn.execute(query, val)
        conn.commit()
        print(data_file , data_clean , sent_dumb)

def cleaning(text):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)    
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)      
    
    return result

if __name__ == "__main__":
    app.run(debug=True)


