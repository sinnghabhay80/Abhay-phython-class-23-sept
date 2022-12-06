import os
from flask import Flask, redirect, render_template, request
import numpy as np
import cv2
import pickle
import pandas as pd
from keras.models import load_model
from keras.utils import img_to_array
from tensorflow import expand_dims
from sklearn.preprocessing import LabelBinarizer


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = load_model('model.hdf5')

default_image_size = tuple((256, 256))

def prediction(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            image = img_to_array(image)
            image = np.array(image, np.float64)
            image = expand_dims(image, axis=0)
            result = model.predict(image)
            result = np.array(result, np.float64)
            prediction = np.where(result==np.max(result))
            return prediction[1][0]
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/about')
def contact():
    return render_template('about-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
