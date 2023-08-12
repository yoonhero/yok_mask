import torch

def convert(token_size, emb_size=100, dir="../checkpoint/word2vec.pt", out="../checkpoint/word2vec.onnx"):
    checkpoint = torch.load(dir)
    for name, param in checkpoint.items():
        if "emb" in name:
            emb = torch.nn.Embedding(token_size, emb_size)
            emb.load_state_dict({"weight": param})
            x = torch.randint(1, token_size, (1, 10))
            torchout = emb(x)

            torch.onnx.export(emb, x, out, export_params=True, do_constant_folding=True, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch", 1: "seq"}})
            break

if __name__ == "__main__":
    convert(50257)

    