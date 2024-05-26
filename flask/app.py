from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('basicheart.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # For now, we are just returning the input as a placeholder for model prediction
    data = request.form.to_dict()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
