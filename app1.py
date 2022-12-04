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





app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False 
SWAGGER_URL = '/swagger'
API_URL = '/static/restapi.yml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Sentiment Analysis"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

conn = sqlite3.connect('record.db', check_same_thread=False)
# conn.execute('''DROP TABLE record;''')
conn.execute('''CREATE TABLE IF NOT EXISTS record(id INTEGER PRIMARY KEY AUTOINCREMENT ,text varchar(255), text_clean varchar(255));''')
# conn.execute('''DROP TABLE record_file;''')

conn.execute('''CREATE TABLE IF NOT EXISTS record_file(id INTEGER PRIMARY KEY AUTOINCREMENT ,text varchar(255), text_clean varchar(255), sentiment varchar(255));''')

MODELNN = joblib.load('mlpc.pkl')
MODELLSTM = load_model('lstmmodelrenewal.h5')





# In[7]:


#welcomepage
@app.route('/', methods=['GET'])
def get():
  return "Presenting API for Sentiment Analysis!"



# In[17]:

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
    sent = MODELNN.predict(tfidf_vectorizer.transform([clean_text]).toarray())
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

    return x
# Function for text cleansing

@app.route('/textLSTM', methods=['POST'])
# TF processing
def text_sentimentTF():

    text = request.form.get('text').lower()
    clean_text = cleaning(text)

    sent = MODELLSTM.predict(tfidf_vectorizerTF.fit_transform([clean_text]).toarray())
    labels = ['negative','neutral', 'positive']
    # sent3 = labels[np.argmax(sent)]
    sent3 = labels[np.argmax(sent[0])]
    query = "INSERT INTO record (text, text_clean) VALUES (?,?)"
    val = (clean_text , str(sent3))
    conn.execute(query , val)
    conn.commit() 

    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': clean_text, 
        'sentiment' : str(sent3),
    }
    response_data = jsonify(json_response)
    return response_data


# Function for csv cleansing
def file_csvNN(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert original text and cleaned text to sqlite database
        data_clean = cleaning(data_file)
        sent = MODELNN.predict(tfidf_vectorizer.transform([data_clean]).toarray()).tolist()
        sent_dumb = json.dumps(sent)
        query = "insert into record_file (text,text_clean ,sentiment) values (?,?,?)"
        val = (data_file, data_clean ,sent_dumb )
        conn.execute(query, val)
        conn.commit()
        print(data_file , data_clean , sent_dumb)
        

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

    # Define API response
    select_data = conn.execute("SELECT * FROM record_file ORDER BY id DESC limit 5")
    conn.commit
    data = [
        dict( id=row[0] , text=row[1], text_clean=row[2] , sentiment = row[3])
    for row  in select_data.fetchall()
    ]
    
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': data, 
        
    }
    response_data = jsonify(json_response)
    return response_data

def file_csvTF(input_file):
    column = input_file.iloc[:, 0]
    print(column)

    for data_file in column: # Define and execute query for insert original text and cleaned text to sqlite database
        data_clean = cleaning(data_file)
        sent = MODELLSTM.predict(tfidf_vectorizerTF.fit_transform([data_clean]).toarray()).tolist()
        labels = ['negative','neutral', 'positive']

        sent3 = labels[np.argmax(sent)]
        # sent_dumb = json.dumps(sent3)
        query = "insert into record_file (text,text_clean ,sentiment) values (?,?,?)"
        val = (data_file, data_clean ,sent3 )
        conn.execute(query, val)
        conn.commit()
        print(data_file , data_clean , sent3)
        

@app.route('/fileLSTM', methods=['POST'])
def file_cleaningTF():

    # Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')
    
    # Cleaning file
    file_csvTF(datacsv)

    # Define API response
    select_data = conn.execute("SELECT * FROM record_file ORDER BY id DESC limit 5 ")
    conn.commit
    data = [
        dict(  id=row[0], text=row[1], text_clean=row[2] , sentiment = row[3])
    for row  in select_data.fetchall()
    ]
    
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': data, 
        
    }
    response_data = jsonify(json_response)
    return response_data


def regex(text):
    text = re.sub('xf', ' ', text)
    text = re.sub('(x\d\w)', ' ', text)
    text = re.sub('\s', ' ', text)
    text = re.sub('rt', ' ', text)
    text = re.sub('user', ' ', text)
    text = re.sub('[^a-zA-Z]+', ' ', text)
    text = re.sub('((www.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub(r'wk[\s]+', ' ', text)
    text = re.sub('0-9', ' ', text)

    
    return text

def lowercase(text):
    return text.lower()

def cleaning(text):
    text = lowercase(text)
    text = regex(text)

    return text

if __name__ == "__main__":
    app.run(debug=True)

#group2
#the end
