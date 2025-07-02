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


name = DataLoader(TextDataset(products['name']), shuffle=False, batch_size=10000)
query = DataLoader(TextDataset(search['query']), shuffle=False, batch_size=10000)


# In[13]:


encoder = Encoder().to(device)


# In[14]:


encoder.load_state_dict(torch.load('encoder.pth'))
encoder.eval()


# In[15]:


def embedd(dl):
    embds = []
    for x in tqdm(dl, leave=False):
        x = x.to(device)
        pred = encoder(x).cpu().detach()
        embds.append(pred)
    embds = torch.concat(embds, dim=0).numpy()
    return embds


# In[16]:


name_e = embedd(name)
products = products.with_columns(pl.Series('name', name_e.tolist()).cast(pl.List(pl.Float32)))
del name_e


# In[17]:


query_e = embedd(query)
search = search.with_columns(pl.Series('query', query_e.tolist()).cast(pl.List(pl.Float32)))
del query_e


# In[18]:


products.write_parquet('product_properties.parquet')
search.write_parquet('search_query.parquet')

