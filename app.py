import sys

import json
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from covidfake.recognize import recognize_entities
from covidfake.detect import detect_fake

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/ner', methods=['POST'])
@cross_origin()
def ner():
    data = json.loads(request.data)
    response = recognize_entities(data['text'])

    return jsonify(response)

@app.route('/detect', methods=['POST'])
@cross_origin()
def predict():
    data = json.loads(request.data)
    response = detect_fake(data['text'])

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)