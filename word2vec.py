import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import tqdm
import random

from dataset import load_all
from tokenizer import Tokenizer

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

def cbow_training(window_size, token_size, emb_size=100, out="./checkpoint/word2vec.pt"):
    emb = nn.Embedding(token_size, emb_size)
    projection = nn.Linear(emb_size, token_size)

    model = nn.Sequential(emb, projection)

    y_loc = window_size//2

    tokenizer = Tokenizer()
    x, _ = load_all()

    x = random.sample(x, 10000)

    tokenized_x = []
    for temp in tqdm.tqdm(x):
        tokenized_x.append(tokenizer.tokenize([temp])[0])
    
    # dataset = []
    x = []
    y = []
    for tokens in tokenized_x:
        if len(tokens) < window_size:
            continue

        for i in range(len(tokens)-window_size-1):
            temp = tokens[i:i+window_size]
            print(temp)
            y_ = [temp[y_loc]]
            temp.pop[y_loc]
            x_ = temp
            x.append(x_)
            y.append(y_)

    # x = torch.tensor(x)
    # y = torch.tensor(y)

    # train_dataset = TensorDataset(x, y)
    train_dataset = SimpleDataSet(x, y)
    train_loader = DataLoader(train_dataset, 64, drop_last=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)

    nb_epochs = 20

    for epoch in range(nb_epochs):
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
            emb_x = emb(x)
            summed_tensor = torch.sum(emb_x, dim=1) / (2*window_size)

            prod = projection(summed_tensor)

            loss = criterion(prod, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"Epoch {epoch}: Train Loss {loss.item():.4f}")

    torch.save(emb.state_dict(), out)


def load_word2vec(token_size, emb_size=100, dir="./checkpoint/word2vec.pt"):
    emb = nn.Embedding(token_size, emb_size)
    emb.load_state_dict(torch.load(dir))

    return emb


if __name__ == "__main__":
    cbow_training(5, 11756, 100)



