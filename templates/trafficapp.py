from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('C:/AI lab/TrafficTelligence Advanced Traffic Volume Estimation with Machine Learning/model.pkl', 'rb'))
scale = pickle.load(open('C:/AI lab/TrafficTelligence Advanced Traffic Volume Estimation with Machine Learning/encoder.pkl', 'rb'))

# Define column names
names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
         'hours', 'minutes', 'seconds']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_values = np.array([input_features])
    data = pd.DataFrame(features_values, columns=names)

    # Scale the data
    scaled_data = scale.transform(data)

    # Predict
    prediction = model.predict(scaled_data)

    text = "Estimated Traffic Volume is: "
    return render_template('index1.html', prediction_text=text + str(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
