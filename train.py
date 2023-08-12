import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import tqdm
import random

from dataset import load_all
# from tokenizer import Tokenizer
import tiktoken

def prepare_dataset(max_len):
# assert enc.decode(enc.encode("hello world")) == "hello world"
    x, y = load_all()
    # x = random.sample(x, 10000)
    indices = random.sample(list(range(len(x))), 10000)
    x = [x[index] for index in indices]
    y = [int(y[index]) for index in indices]

    # tokenized_x = []
    # for temp in tqdm.tqdm(x):
        # tokenized_x.append(tokenizer.tokenize([temp])[0])

    train_dataset = SimpleDataSet(x, y, max_len)
    train_loader = DataLoader(train_dataset, 64, drop_last=True)
    
    return train_loader

class SimpleDataSet(Dataset):
    def __init__(self, x, y, max_len):
        super().__init__()
        self.enc = tiktoken.get_encoding("r50k_base")
        self.x = x
        self.y = y
        self.max_len = max_len
    
    def __getitem__(self, idx):
        t = self.x[idx]
        tokens = self.enc.encode(t)
        if len(tokens) < self.max_len:
            tokens = tokens + [50256] * (self.max_len-len(tokens))
        tokens = tokens[:self.max_len]
        _x = torch.tensor(tokens)
        _y = torch.tensor(self.y[idx])

        return _x, _y

    def __len__(self):
        return len(self.x)
    
class SimpleNN(nn.Module):
    def __init__(self, token_size, emb_size, seq_len):
        super().__init__()
        self.emb = nn.Embedding(token_size, emb_size)
        self.linear1 = nn.Linear(seq_len*emb_size, token_size)
        self.linear2 = nn.Linear(token_size, 5000)
        self.linear3 = nn.Linear(5000, 1)
        # self.linear3.weight.data /= 1e+2
        self.relu = nn.ReLU()
        self.linears = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T = x.shape
        emb_x = self.emb(x)
        emb_x = emb_x.view(B, -1)
        prod = self.linears(emb_x)
        return prod

def train(token_size, emb_size=100, out="./checkpoint/word2vec.pt"):
    max_len = 15
    train_loader = prepare_dataset(max_len)
    model = SimpleNN(token_size=token_size, emb_size=emb_size, seq_len=max_len)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.95))

    nb_epochs = 100

    for epoch in range(nb_epochs):
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
            prod = model(x)   
            prod = torch.squeeze(prod, 1)
            loss = criterion(prod.to(torch.float32), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch}: Train Loss {loss.item():.4f}")

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f"./checkpoint/cbow_{epoch}.pt")


def load_word2vec(token_size, emb_size=100, dir="./checkpoint/word2vec.pt"):
    emb = nn.Embedding(token_size, emb_size)
    emb.load_state_dict(torch.load(dir))

    return emb

if __name__ == "__main__":
    train(50257, 100)



