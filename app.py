import pickle,os
from flask import Flask, request, render_template
from flask_restful import Api

with open("RTA_Model_pickle.pkl", 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__,template_folder='templates')
api = Api(app)

@app.route('/')
def start():
    return 'Praca domowa Łukasz Makaruk :D'

@app.route('/api/predict/')
def my_form():
    return render_template('index.html')
@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"Brak danych do predykcji, aby je uzupelnic przejdz do /api/predict"
    if request.method == 'POST':
        form_data = request.form
        pl=form_data['pl']
        sl=form_data['sl']
        res = model.predict([float(sl), float(pl)])
        mapper = {'0': 'Setosa',
              '1': 'Versicolor'}
        return mapper[f"{res}"]

app.run(port='5032',host='0.0.0.0')

