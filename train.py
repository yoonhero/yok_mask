import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import tqdm
import random
import matplotlib.pyplot as plt

from dataset import load_all
# from tokenizer import Tokenizer
import tiktoken

def prepare_dataset(max_len):
# assert enc.decode(enc.encode("hello world")) == "hello world"
    x, y = load_all()
    # x = random.sample(x, 10000)
    indices = random.sample(list(range(len(x))), 200000)
    x = [x[index] for index in indices]
    y = [int(y[index]) for index in indices]

    # tokenized_x = []
    # for temp in tqdm.tqdm(x):
        # tokenized_x.append(tokenizer.tokenize([temp])[0])

    train_dataset = SimpleDataSet(x[:-1000], y[:-1000], max_len)
    eval_dataset = SimpleDataSet(x[-1000:], y[-1000:], max_len)
    train_loader = DataLoader(train_dataset, 2048, drop_last=True)
    eval_loader = DataLoader(eval_dataset, 256)
    
    return train_loader, eval_loader

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

        return _x.to("cuda"), _y.to("cuda")

    def __len__(self):
        return len(self.x)
    
class SimpleNN(nn.Module):
    def __init__(self, token_size, emb_size, seq_len):
        super().__init__()
        self.emb = nn.Embedding(token_size, emb_size)
        self.linear1 = nn.Linear(seq_len*emb_size, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 1)
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
    max_len = 50
    train_loader, eval_loader = prepare_dataset(max_len)
    model = SimpleNN(token_size=token_size, emb_size=emb_size, seq_len=max_len).to("cuda")
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95))

    nb_epochs = 50

    train_loss = []
    eval_acc = []
    for epoch in range(nb_epochs):
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
            prod = model(x)   
            prod = torch.squeeze(prod, 1)
            loss = criterion(prod.to(torch.float32), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print("eval start")
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for (x, y) in eval_loader:
                prod = model(x)   
                prod = torch.squeeze(prod, 1)
                prediction = (prod > 0.5).long()
                num_correct += (prediction == y).sum()
                num_samples += prediction.size(0)

        train_loss.append(loss.item())
        eval_acc.append(num_correct/num_samples)
        print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Eval Acc {eval_acc[-1]}")

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f"./checkpoint/train_{epoch}.pt")

    x_range = list(range(len(train_loss)))
    plt.plot(x_range, train_loss)
    plt.show()
    plt.plot(x_range, eval_acc)
    plt.show()

if __name__ == "__main__":
    # train(50257, 100)

    enc = tiktoken.get_encoding("r50k_base")
    model = SimpleNN(50257, 100, 50)
    model.load_state_dict(torch.load("checkpoint/train_49.pt"))

    tokens = enc.encode("어릴 때 보고 지금 다시 봐도 재밌어요ㅋㅋ")[:50]
    print(enc.decode(tokens), tokens)
    tokens = tokens + [50256] * (50 -len(tokens))
    tokens = torch.tensor(tokens)
    tokens = tokens.unsqueeze(0)

    print(model(tokens))
    

    



