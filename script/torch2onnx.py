import torch

def convert(token_size, emb_size=100, dir="../checkpoint/word2vec.pt", out="../checkpoint/word2vec.pt"):
    emb = torch.nn.Embedding(token_size, emb_size)
    emb.load_state_dict(torch.load(dir))
    x = torch.randint(1, 11756, (1, 10))
    torchout = emb(x)

    torch.onnx.export(emb, x, out, export_params=True, do_constant_folding=True, input_names=["input"], output_names=["output"])

if __name__ == "__main__":
    convert()