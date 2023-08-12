from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import numpy as np
import tiktoken
import random
import re

import onnxruntime
import numpy as np

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
enc = tiktoken.get_encoding("r50k_base")
ort_session = onnxruntime.InferenceSession("word2vec.onnx")

def preprocess(text):
    idx = enc.encode(text)
    input = np.array([idx])
    ort_inputs = {"input": input}
    return idx, ort_inputs

def predict(text):
    idx, input = preprocess(text)
    ort_outs = ort_session.run(None, input)
    embed = ort_outs[0][0, :, :] # (seq_len, 100)

    return idx, embed

yok = []
with open("fword_list.txt", "r", encoding="utf-8") as f:
    data = f.read()
    yok = data.split("\n")

yok_vec = []
for y in yok:
    _, emb = predict(y)
    concat_emb = np.sum(emb, 0)
    yok_vec.append(concat_emb)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))

def get_yok_location(text):
    idx, embed = predict(text)
    selected_yok = random.sample(yok_vec, 1)[0]

    sim = cosine_sim(embed, selected_yok)
    to_mask = (sim > 0.3).astype(np.int8).tolist()
    print(sim)
    return idx, to_mask

@app.route('/', methods=['POST'])
def upload_train():
    data = request.get_json()
    base_mask = "😷"
    try:
        mask_text = data["to"] if data["to"] != None else base_mask
    except: mask_text = base_mask
    encoded_mask = " ".join([str(idx) for idx in enc.encode(mask_text)])
    print(data["text"])

    idx, masking_loc = get_yok_location(data['text'])
    for i, loc in enumerate(masking_loc):
        if loc == 1:
            idx[i] = encoded_mask
    idx = [int(i) for i in " ".join([str(i) for i in idx]).split(" ")]
    result = enc.decode(idx)
    result = result.replace("�", "")
    # result = re.sub(r'[^\u0020-\ud7ff]', '', result)

    response = Response()
    response.headers[
        'Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin, X-Requested-With, Content-Type, Accept, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, HEAD'
    
    print(result)

    return jsonify({"result":result}), 200

if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True, threaded=True)