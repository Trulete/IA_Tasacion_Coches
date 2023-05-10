from flask import Flask, request, Response, render_template
import pickle
from flask_cors import CORS
import json
import tensorflow as tf
import pandas as pd

application = Flask(__name__)
CORS(application)



@application.route('/flask', methods=['GET'])
def flask():
    return '<p>https://flask.palletsprojects.com/en/2.2.x/</p>'

@application.get('/api/get')
def get_method():
    word = request.args.get('word', '<no word>')
    return {
        'hello': 'hello, ' + word
    }

@application.route('/')
def main():
    return '<p>Hello, World!</p>'

@application.route('/polo')
def polo():
    with open('modelo_reg_ohe.pck', 'rb') as file:
        dv, model = pickle.load(file)
    coche = {
        'marca': 'vw',
        'modelo': 'polo',
        'cv': 110
    }
    coche_bien_codificado = dv.transform(coche)
    precio = model.predict(coche_bien_codificado)
    return {
        'precio': precio[0]
    }

@application.route('/cars')
def car2():
    #with open('modelo_reg_ohe.pck', 'rb') as file:
    #    dv, model = pickle.load(file)
    with open('modelo_coches_net.pck', 'rb') as file:
        dv, model = pickle.load(file)
    marca = request.args.get('marca', '')
    modelo = request.args.get('modelo', '0')
    fuelType = request.args.get('fuelType', '')
    province = request.args.get('province', '')
    color = request.args.get('color', '')
    transmissionType = request.args.get('transmissionType', '')
    bodyType = request.args.get('bodyType', '')
    doors_str = request.args.get('doors', '')
    doors = int(doors_str)
    cv_str = request.args.get('cv', '')
    cv = int(cv_str)
    km_str = request.args.get('km', '')
    km = int(km_str)
    cc_str = request.args.get('cubicCapacity', '')
    cc = int(cc_str)

    coche = {
        'brand': marca,
        'model': modelo,
        'fuelType': fuelType,
        'province': province,
        'color': color,
        'transmissionType': transmissionType,
        'bodyType': bodyType,
        'doors': doors,
        'cv': cv,
        'km': km,
        'cubicCapacity': cc
    }
    coche_bien_codificado = dv.transform(coche)
    try:
        precio = model.predict(coche_bien_codificado)
    except:
        precio = 0
    return {
        'precio': precio[0]
    }

@application.route('/home')
def home():
    return render_template('index.html')

@application.get('/car')
def car():
    neural_model = tf.keras.models.load_model('modelo_coches_net.hdf5')
    
    with open('dv_coches_net.pck', 'rb') as file:
     dv = pickle.load(file)

    campos_numericos = ['km', 'year', 'cubicCapacity', 'doors', 'hp']

    # obtenemos el coche de la request

    coche = json.loads(request.args.get('coche', ''))
    coche['year'] = int(coche['year'])
    coche['hp'] = int(coche['hp'])
    coche['km'] = int(coche['km'])
    coche['doors'] = float(coche['doors'])
    coche['cubicCapacity'] = int(coche['cubicCapacity'])
    coche_df = pd.DataFrame(coche, index=[0])
    coche_dv = dv.transform(coche_df.to_dict(orient='records'))
    
    try:
        precio = neural_model.predict(coche_dv).tolist()
    except:
        precio = [0]
    
    print(precio)

    return {
       'precio': precio[0]
    }


application.run()