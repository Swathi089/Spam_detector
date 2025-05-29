# app.py

from flask import Flask, render_template, request
from spam_detector import predict_spam

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']
    result = predict_spam(email)
    return render_template('index.html', email=email, result=result)

if __name__ == '__main__':
    app.run(debug=True)
