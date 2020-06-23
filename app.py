import sys
sys.path.insert(1, '/home/jdnietov/Development/uni/nlp/project/api/services')

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from ner import recognize_entities

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

recognize_entities('')

@app.route('/ner')
@cross_origin()
def hello():
    body = recognize_entities('')

    return jsonify(body)

if __name__ == "__main__":
    app.run(debug=True)