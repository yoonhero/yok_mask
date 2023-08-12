import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import tqdm
import random

from dataset import load_all
from tokenizer import Tokenizer

def prepare_dataset(window_size):
    y_loc = window_size // 2

    tokenizer = Tokenizer()
    x, _ = load_all()

    x = random.sample(x, 10000)

    tokenized_x = []
    for temp in tqdm.tqdm(x):
        tokenized_x.append(tokenizer.tokenize([temp])[0])

    x = []
    y = []
    for tokens in tokenized_x:
        if len(tokens) < window_size:
            continue

        for i in range(len(tokens)-window_size-1):
            temp = tokens[i:i+window_size]
            if temp == [0] * window_size:
                continue
            y_ = temp[y_loc]
            x_ = temp[:y_loc] + temp[y_loc+1:]
            x.append(x_)
            y.append(y_)

    train_dataset = SimpleDataSet(x, y)
    train_loader = DataLoader(train_dataset, 64, drop_last=True)
    
    return train_loader

class SimpleDataSet(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        _x = torch.tensor(self.x[idx])
        _y = torch.tensor(self.y[idx])

        return _x, _y

    def __len__(self):
        return len(self.x)
    
class SimpleCBOW(nn.Module):
    def __init__(self, token_size, emb_size, window_size):
        super().__init__()
        self.emb = nn.Embedding(token_size, emb_size)
        self.projection = nn.Linear(emb_size, token_size)
        self.window_size = window_size

    def forward(self, x):
        emb_x = self.emb(x)
        summed_tensor = torch.sum(emb_x, dim=1) / (2*self.window_size)

        prod = self.projection(summed_tensor)
        return prod

def cbow_training(window_size, token_size, emb_size=100, out="./checkpoint/word2vec.pt"):
    model = SimpleCBOW(token_size=token_size, emb_size=emb_size, window_size=window_size)
    
    train_loader = prepare_dataset(window_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95))

    nb_epochs = 100

    for epoch in range(nb_epochs):
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
            prod = model(x)   

            loss = criterion(prod, y)
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
    cbow_training(5, 11757, 100)



