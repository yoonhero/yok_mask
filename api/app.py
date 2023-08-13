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
    # print(input["input"].shape,ort_outs[0].shape)

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

def cosine_sim(a, b, eps=1e-4):
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + eps)

def word_trim(idx, emb):
    new_idx = []
    breakwords = [220, 23821]

    temp = []
    for i, token in enumerate(idx):
        if i == len(idx) - 1:
            temp.append(token)
            new_idx.append(temp)
            break
        if token in breakwords:
            new_idx.append(temp)
            new_idx.append([token])
            temp = []
            continue
        temp.append(token)
    tmp = 0
    temp_embs = []
    for trim in new_idx:
        length = len(trim)
        sum_emb = np.sum(emb[tmp:tmp+length], axis=0)
        temp_embs.append(sum_emb)
        tmp += length
    new_emb = np.stack(temp_embs)

    return new_idx, new_emb
    

def get_yok_location(text, min_sim):
    idx, embed = predict(text)
    new_idx, new_emb = word_trim(idx, embed)

    # selected_yok = random.sample(list(range(len(yok_vec))), 1)[0]
    random_idx = random.randint(0, len(yok_vec)-1)
    print(yok[random_idx])
    selected_yok = yok_vec[random_idx]

    sim = cosine_sim(new_emb, selected_yok)
    to_mask = (sim > min_sim).astype(np.int8).tolist()
    print(sim)

    tmp = 0
    tmp_mask = [0] * len(idx)
    for i, m in enumerate(to_mask):
        idx_set = new_idx[i]
        if m == 1:
            tmp_mask = tmp_mask[:tmp] + [1]*len(idx_set) + tmp_mask[tmp+len(idx_set):]
            # for i in range(len(idx_set)):
            #     tmp_mask[tmp+i] = 1
        tmp += len(idx_set)
    return idx, tmp_mask, sim.tolist()

@app.route('/chk', methods=['POST'])
def upload_train():
    data = request.get_json()
    base_mask = "😷"
    try:
        mask_text = data["to"] if data["to"] != None else base_mask
    except: mask_text = base_mask
    encoded_mask = " ".join([str(idx) for idx in enc.encode(mask_text)])
    try:
        min_sim = float(data["min_sim"])
    except: min_sim = 0.5
    print(data["text"])

    idx, masking_loc, similarity = get_yok_location(data['text'], min_sim)
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

    # return jsonify({"result":result, "sim": similarity}), 200
    return jsonify({"result": result}), 200

if __name__ == "__main__":
    app.run(port=5555, host='0.0.0.0', debug=True, threaded=True)
