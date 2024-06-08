
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('../Exploratory Data Analysis/bagged_decision_tree_model.pkl')
scaler = joblib.load('../Exploratory Data Analysis/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print(data)
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca']
    
    # Convert input data to a dataframe
    input_data = pd.DataFrame([data], columns=features)
    
    # Convert types to float
    input_data = input_data.astype(float)
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Return the result as JSON
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
