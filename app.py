from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('data/processed/feature_scaler.pkl')

# Feature names (same order as used during preprocessing)
FEATURE_NAMES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation',
    'AgriculturalPractices', 'Encroachments', 'IneffectiveDisasterPreparedness',
    'DrainageSystems', 'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

@app.route('/')
def home():
    return render_template('form.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input for each feature
        inputs = []
        for name in FEATURE_NAMES:
            value = float(request.form[name])
            inputs.append(value)

        # Convert to array and scale
        features = np.array([inputs])
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_features)[0]
            confidence = max(proba)
        else:
            confidence = None

        classes = ['Low', 'Medium', 'High']
        response = f"{classes[prediction]}"
        if confidence is not None:
            response += f" (Confidence: {confidence:.2%})"

        return render_template('predict.html', result=response)


    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
