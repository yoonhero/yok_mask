from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np

from tokenizer import Tokenizer


app = Flask(__name__)
CORS(app)

import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("word2vec.onnx")

def word2vec(idx):
    input = np.array(idx)
    ort_inputs = {"input": input}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs["output"]

@app.route('/', methods=['POST'])
def upload_train():
    data = request.get_json()

    poornag = word2vec(data['text'])

    response = Response()
    response.headers[
        'Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin, X-Requested-With, Content-Type, Accept, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, HEAD'

    return poornag, 200

if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True, threaded=True)