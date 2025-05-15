from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and data
model = joblib.load('no_show_model.pkl')
df = pd.read_csv('appointments.csv')
features = ['Age', 'SMS_Received', 'Weekday', 'Previous_NoShows', 'Diabetes', 'Hypertension', 'Scholarship']

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def home():
    stats = {
        'total_appointments': len(df),
        'no_show_rate': round(df['NoShow'].mean() * 100, 1),
        'no_show_with_sms': round(df[df['SMS_Received'] == 1]['NoShow'].mean() * 100, 1),
        'no_show_without_sms': round(df[df['SMS_Received'] == 0]['NoShow'].mean() * 100, 1),
        'weekday_rates': df.groupby('Weekday')['NoShow'].mean().round(3).to_dict()
    }
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': int(request.form['age']),
        'SMS_Received': int(request.form['sms_received']),
        'Weekday': int(request.form['weekday']),
        'Previous_NoShows': int(request.form['previous_noshows']),
        'Diabetes': int(request.form.get('diabetes', 0)),
        'Hypertension': int(request.form.get('hypertension', 0)),
        'Scholarship': int(request.form.get('scholarship', 0))
    }
    
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    probability = model.predict_proba(pd.DataFrame([input_data]))[0][1]
    
    result = {
        'prediction': 'High No-Show Risk' if prediction == 1 else 'Low No-Show Risk',
        'probability': round(probability * 100, 1),
        'input_data': input_data
    }
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  