# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:05:14 2019

@author: f'ewf'
"""

import os
import numpy as np
import flask
import pickle
from keras import backend as k
from keras.preprocessing.sequence import pad_sequences
from flask import Flask,render_template,request

k.clear_session()
app=Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def generate_text(seq_len,seed_text,num_gen_words):
    
    output_text=[]
    model=pickle.load(open("model.pkl","rb"))
    tokenizer=pickle.load(open('my_simple_tokenizer','rb'))
    
    input_text=seed_text
    for i in range(num_gen_words):
        
        encoded_text=tokenizer.texts_to_sequences([input_text])[0]
        
        pad_encoded=pad_sequences([encoded_text,],maxlen=seq_len,truncating='pre')
        
        pred_word_ind=model.predict_classes(pad_encoded,verbose=0)[0]
        
        pred_word=tokenizer.index_word[pred_word_ind]
        
        input_text+=' '+pred_word
        
        output_text.append(pred_word)
    return ' '.join(output_text)



@app.route('/result',methods=['POST'])
def result():
    if request.method=='POST':
        to_predict_list=request.form['text']
       # to_predict_list=to_predict_list.upper()
        k.clear_session()
        result=generate_text(25,to_predict_list,10)
        
        k.clear_session()
        return render_template("index.html",prediction_text=result)
    
    