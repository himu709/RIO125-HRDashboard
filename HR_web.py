# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:12:57 2021

@author: Himesh
"""

from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
      return render_template('welcome.html')
@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output =prediction[0]

    return render_template('welcome.html', prediction_text='Your Predicted Salary is {}'.format(output))
if __name__== '__main__':
    app.run(port=8000)