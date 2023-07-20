from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)
model = pd.read_pickle('car_predictor.pkl')
app.template_folder = 'template'
data = pd.read_csv('cleaned_car.csv')


@app.route('/', methods=['GET'])
def index():
    companies = sorted(data['company'].unique())
    car_models = sorted(data['name'].unique())
    year = sorted(data['year'].unique(), reverse=True)
    fuel_type = data['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('modal')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('km')
    print(company, car_model, year, fuel_type, driven)
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array(
        [car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))


# run the app.
if __name__ == "__main__":
    app.run(debug=True)
