import pandas as pd
from flask import Flask, render_template, request
import joblib
import os
import sklearn.ensemble._forest
app2 = Flask(__name__)

# Load model and feature order
model = joblib.load("C:/AI lab/TrafficTelligence Advanced Traffic Volume Estimation with Machine Learning/model.joblib")
feature_order = joblib.load("C:/AI lab/TrafficTelligence Advanced Traffic Volume Estimation with Machine Learning/feature_order.joblib")

@app2.route('/')
def home():
    return render_template('index1.html')

@app2.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # Prepare numeric inputs (excluding traffic_volume)
        input_values = {
            'rain': float(form['rain']),
            'snow': float(form['snow']),
            'day': int(form['day']),
            'month': int(form['month']),
            'year': int(form['year']),
            'hours': int(form['hours']),
            'minutes': int(form['minutes']),
            'seconds': int(form['seconds']),
        }

        # Initialize input dictionary with zeroed features
        input_data = {col: 0 for col in feature_order}
        input_data.update(input_values)

        # One-hot encode holiday and weather
        holiday_feature = f"holiday_{form['holiday']}"
        weather_feature = f"weather_{form['weather']}"

        if holiday_feature in input_data:
            input_data[holiday_feature] = 1
        if weather_feature in input_data:
            input_data[weather_feature] = 1

        # Create DataFrame for prediction
        final_input = pd.DataFrame([input_data])

        # Predict traffic volume
        prediction = model.predict(final_input)[0]
        estimated_volume = round(prediction, 2)

        return render_template('result.html', prediction=estimated_volume)

    except Exception as e:
        return f"Error in prediction logic: {str(e)}"

if __name__ == '__main__':
    app2.run(debug=True, use_reloader=False)