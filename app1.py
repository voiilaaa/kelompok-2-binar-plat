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
# import pandas as pd

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

MODEL = joblib.load('mlpc.pkl')
# MODEL_LABELS = ['tweet', 'label']




# In[7]:


#welcomepage
@app.route('/', methods=['GET'])
def get():
  return "Presenting API for Sentiment Analysis!"



# In[17]:

# @swag_from("docs/hint.yml", methods=['POST'])
tfidf_vectorizer = joblib.load('feature.pkl') 

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


@app.route('/fileNN', methods=['POST'])
def upload_fileNN():

    file = request.files['file']
    data = pd.read_csv(file, encoding='iso-8859-1')
    data_list = data.values.tolist()
    clean_data_list = []
    # print (column)

    for i in data_list:
        clean_data_list.append(re.sub(r'[^a-z0-9]',' ',data_list[i]))
        query_text = "insert into data(text,text_clean) values(?,?)"
        val = (data,clean_data_list)
        conn.execute(query_text , val)
        conn.commit

    #Define response API
    json_response={
        'status_code': 200,
        'description': 'text cleanse',
        'text': [clean_data_list], 
        # 'sentiment' : (label),
        
    }
    response_data = jsonify(json_response)
    return response_data



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


