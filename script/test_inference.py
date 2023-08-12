import tiktoken
import numpy as np
import onnxruntime

enc = tiktoken.get_encoding("r50k_base")
ort_session = onnxruntime.InferenceSession("../checkpoint/word2vec.onnx")

tokens = enc.encode("어릴 때 보고 지금 다시 봐도 재밌어요ㅋㅋ")
input = np.array([tokens])
ort_inputs = {"input": input}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0][0, :, :])

    



