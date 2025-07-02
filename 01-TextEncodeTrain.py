#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import numpy as np
import os
from datetime import datetime
from tqdm.auto import tqdm
from pymongo import MongoClient
import pymongo
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score


# In[2]:


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# In[3]:


def to_int_list(x):
    x = x[1:-1].split(' ')
    x = [int(i) for i in x if i != '']
    return x


# In[4]:


products = pl.read_parquet('data/product_properties.parquet')
search = pl.read_parquet('data/search_query.parquet')


# In[5]:


search = search.with_columns(pl.col('query').map_elements(to_int_list, return_dtype=pl.List(pl.Int64)))
products = products.with_columns(pl.col('name').map_elements(to_int_list, return_dtype=pl.List(pl.Int64)))


# In[6]:


print(len(search))
print(len(products))


# In[7]:


class TextDataset(Dataset):

    def __init__(self, col):
        self.data = np.asarray([list(x) for x in col])
        self.data = torch.from_numpy(self.data).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :]


# In[8]:


class Encoder(nn.Module):

    def __init__(self, tokens=256, dims=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(tokens, dims)
        self.layers = nn.LSTM(dims, dims, num_layers=layers, batch_first=True, bidirectional=True)
        self.final = nn.Linear(2 * dims, dims)
        pass

    def forward(self, x):
        x = self.emb(x)
        x = self.layers(x)[0][:, -1, :]
        x = self.final(x)
        x = torch.tanh(x)
        return x

class Decoder(nn.Module):

    def __init__(self, length=16, tokens=256, dims=64, layers=2):
        super().__init__()
        self.length = length
        self.layers = nn.LSTM(dims, dims, num_layers=layers, batch_first=True, bidirectional=True)
        self.final = nn.Linear(2 * dims, tokens)
        pass

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.length, 1)
        x = self.layers(x)[0]
        x = self.final(x)
        return x


# In[9]:


Encoder()(torch.ones((3, 16)).long()).shape


# In[10]:


Decoder()(Encoder()(torch.ones((3, 16)).long())).shape


# In[11]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[12]:


train = DataLoader(TextDataset(products['name']), shuffle=True, batch_size=2048)
test = DataLoader(TextDataset(search['query']), batch_size=10000)


# In[13]:


encoder = Encoder().to(device)
decoder = Decoder().to(device)


# In[14]:


def eval(cl_report=False):
    real = []
    pr = []
    encoder.eval()
    decoder.eval()
    for x in tqdm(test, leave=False):
        x = x.to(device)
        pred = decoder(encoder(x))
        m = pred.argmax(dim=-1)
        real.append(x.cpu().view(-1).numpy())
        pr.append(m.detach().cpu().view(-1).numpy())
    real = np.concatenate(real)
    pr = np.concatenate(pr)
    print(f'Acc {accuracy_score(real, pr) * 100:.2f} %')
    print(f'Balanced Acc {balanced_accuracy_score(real, pr) * 100:.2f} %')
    if cl_report:
        print(classification_report(real, pr))
    


# In[15]:


criteria = nn.CrossEntropyLoss()
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))


# In[16]:


for e in range(1, 100):
    encoder.train()
    decoder.train()
    for x in tqdm(train, desc=f'Epoch: {e:3d}', leave=False):
        x = x.to(device)
        opt.zero_grad()
        pred = decoder(encoder(x))
        loss = criteria(pred.view(-1, 256), x.view(-1))
        loss.backward()
        opt.step()
        pass
    if e % 10 == 0:
        eval()


# In[17]:


eval(True)


# In[18]:


torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')

