from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import json
import xgboost as xgb
import pickle
import os

app = Flask(__name__)

# Load the model
def load_model():
    try:
        # Try to load as XGBoost model from JSON
        model = xgb.Booster()
        model.load_model('model.json')
        model_type = "xgboost"
        return model, model_type
    except:
        try:
            # Try to load as pickle file (scikit-learn or other)
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            model_type = "sklearn"
            return model, model_type
        except:
            return None, None

model, model_type = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        features = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'total sulfur dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['ph']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }
        
        # Create DataFrame from features
        input_df = pd.DataFrame([features])
        
        # Make prediction
        if model_type == "xgboost":
            dmatrix = xgb.DMatrix(input_df)
            prediction = model.predict(dmatrix)
        else:  # sklearn
            prediction = model.predict(input_df)
        
        # Get quality value
        quality = float(prediction[0])
        
        # Determine quality category
        if quality >= 7:
            category = "Excellent"
        elif quality >= 5:
            category = "Good"
        else:
            category = "Below Average"
        
        return render_template('result.html', 
                              quality=round(quality, 2), 
                              category=category,
                              features=features)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
