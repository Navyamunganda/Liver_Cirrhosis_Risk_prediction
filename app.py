from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('rf_acc_68.pkl', 'rb') as f:
    model = pickle.load(f)
with open('normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['age']),
        float(request.form['gender']),
        float(request.form['total_bilirubin']),
        float(request.form['alk_phos']),
        float(request.form['sgpt']),
        float(request.form['sgot']),
        float(request.form['total_proteins']),
        float(request.form['albumin']),
        float(request.form['ag_ratio'])
    ]
    input_array = np.array([input_data])
    input_scaled = normalizer.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    return render_template('inner-page.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)