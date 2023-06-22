from flask import Flask, jsonify, render_template, request, make_response
import transformers
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

model = pipeline(model='deandrasetya/indobert-abusive-language-classifier')

def classify(message):
    result = model(message)
    
    status = 'SAFE'

    if result[0]['label'] == 'ABUSIVE':
        toxicity_level = '{:.1%}'.format(result[0]['score'])

        if result[0]['score'] >= 0.7:
            status = 'ABUSIVE'
    else:
        toxicity_level = '{:.1%}'.format(1 - result[0]['score'])

    prediction = {
        "status": status,
        "score": toxicity_level
    }

    return prediction


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction = classify(message)
    response = jsonify(prediction)

    return response
    

if __name__ == '__main__':
    # starting app
    app.run(host='0.0.0.0')