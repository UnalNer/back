import sys

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from covidfake.recognize import recognize_entities
from covidfake.detect import detect_fake

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

recognize_entities('')
detect_fake('lala')

@app.route('/ner')
@cross_origin()
def hello():
    body = recognize_entities('')

    return jsonify(body)

if __name__ == "__main__":
    app.run(debug=True)