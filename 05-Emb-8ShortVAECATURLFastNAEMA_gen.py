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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
from scipy import stats
from functools import partial
from torch.nn.utils.rnn import pad_sequence

# In[2]:


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# In[3]:


print(torch.cuda.get_device_name())
print(torch.cuda.mem_get_info())


# In[4]:


df = pl.read_parquet('full_enc_search.parquet')
cl = np.load('data/input/relevant_clients.npy')
df = df.filter(pl.col('client_id').is_in(set(cl)))
df.head()
del cl


# In[5]:


prod_ids = set()
for x in tqdm(df.filter(pl.col('sku_add').is_not_null())['sku_add']):
    prod_ids.update(x)
for x in tqdm(df.filter(pl.col('sku_rm').is_not_null())['sku_rm']):
    prod_ids.update(x)
for x in tqdm(df.filter(pl.col('sku_buy').is_not_null())['sku_buy']):
    prod_ids.update(x)
prod_ids = list(prod_ids)
prod_ids.sort()
map_prod_id = {}
for i, n in enumerate(tqdm(prod_ids)):
    map_prod_id[n] = i


# In[6]:


#Check client ids
client_ids = list(set(df['client_id']))
client_ids.sort()
map_client_id = {}
for i, n in enumerate(tqdm(client_ids)):
    map_client_id[n] = i


# In[7]:


urls = set()
for x in tqdm(df.filter(pl.col('url_visit').is_not_null())['url_visit']):
    urls.update(x)
urls = list(urls)
urls.sort()
map_urls_id = {}
for i, n in enumerate(tqdm(urls)):
    map_urls_id[n] = i


products = pl.read_parquet('data/product_properties.parquet')
products = products.filter(pl.col('sku').is_in(set(prod_ids)))
cat = list(set(products['category'].to_list()))
cat.sort()
map_cats_id = {}
for i, c in enumerate(tqdm(cat)):
    map_cats_id[c] = i

map_prod_cat_id = {}
for s in products.select('sku', 'category').iter_rows():
    map_prod_cat_id[s[0]] = map_cats_id[s[1]]

# In[8]:


print(f'Clients: {len(map_client_id)}')
print(f'Urls: {len(map_urls_id)}')
print(f'Prods: {len(map_prod_id)}')


# In[9]:


del urls
del prod_ids
del cat


# In[10]:


# Define the Gaussian kernel function
def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

# Kernel Density Estimation (KDE)
def kde(data, x_grid, bandwidth):
    n = len(data)
    kde_values = np.zeros(len(x_grid), dtype=np.float32)
    if data is None:
        return kde_values
    # Compute the density estimate at each point in x_grid
    for i, x in enumerate(x_grid):
        kernel_sum = 0
        for xi in data:
            kernel_sum += gaussian_kernel((x - xi) / bandwidth)
        kde_values[i] = kernel_sum / (n * bandwidth)
    
    return kde_values


# In[11]:


bandwidth = 0.1
points = np.arange(11) / 10

kde_func = partial(kde, x_grid=points, bandwidth=bandwidth)





print('Start processing!')
if not os.path.exists('enc2_fast_cat.parquet'):
    df = df.with_columns(
        pl.col('client_id').replace_strict(map_client_id),
        pl.col('url_visit').map_elements(
            lambda x: None if x is None else [map_urls_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)),
        pl.col('sku_buy').map_elements(
            lambda x: None if x is None else [map_prod_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)),
        pl.col('sku_rm').map_elements(
            lambda x: None if x is None else [map_prod_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)),
        pl.col('sku_add').map_elements(
            lambda x: None if x is None else [map_prod_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)),
        #Categories
        pl.col('sku_buy').map_elements(
            lambda x: None if x is None else [map_prod_cat_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)).alias('sku_buy_cat'),
        pl.col('sku_rm').map_elements(
            lambda x: None if x is None else [map_prod_cat_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)).alias('sku_rm_cat'),
        pl.col('sku_add').map_elements(
            lambda x: None if x is None else [map_prod_cat_id[e] for e in x],
            return_dtype=pl.List(pl.Int64)).alias('sku_add_cat'),
        pl.col('query_search').map_elements(
             lambda x: None if x is None else stats.mode(x.to_list(), axis=0).mode,
             return_dtype=pl.List(pl.Float32))
    )
    print('Second pass')
    df = df.with_columns(pl.col('ts_proportional_visit').list.len().alias('visits'), 
                    pl.col('ts_proportional_add').list.len().alias('adds'), 
                    pl.col('ts_proportional_rm').list.len().alias('rms'), 
                    pl.col('ts_proportional_search').list.len().alias('searchs'), 
                    pl.col('ts_proportional_buy').list.len().alias('buys'),
                    pl.col('ts_proportional_visit').map_elements(kde_func, return_dtype=pl.List(pl.Float32)).alias('kde_visits'), 
                    pl.col('ts_proportional_add').map_elements(kde_func, return_dtype=pl.List(pl.Float32)).alias('kde_adds'), 
                    pl.col('ts_proportional_rm').map_elements(kde_func, return_dtype=pl.List(pl.Float32)).alias('kde_rms'), 
                    pl.col('ts_proportional_search').map_elements(kde_func, return_dtype=pl.List(pl.Float32)).alias('kde_searchs'), 
                    pl.col('ts_proportional_buy').map_elements(kde_func, return_dtype=pl.List(pl.Float32)).alias('kde_buys')
                    )
    df = df.select(['client_id', 'visits', 'adds', 'rms', 'searchs', 'buys',
                    'kde_visits', 'kde_adds', 'kde_rms', 'kde_searchs', 'kde_buys',
                    'url_visit', 'sku_add', 'sku_rm', 'query_search', 'sku_buy',
                    'sku_add_cat', 'sku_rm_cat', 'sku_buy_cat'])
    df.write_parquet('enc2_fast_cat.parquet')
else:
    print('Loading')
    del df
    df = pl.read_parquet('enc2_fast_cat.parquet')
    pass

print('Done!')





df = df.with_columns(
    pl.col('visits').fill_null(0),
    pl.col('adds').fill_null(0),
    pl.col('rms').fill_null(0),
    pl.col('searchs').fill_null(0),
    pl.col('buys').fill_null(0),
    pl.col('kde_visits').fill_null([0] * 11),
    pl.col('kde_adds').fill_null([0] * 11),
    pl.col('kde_rms').fill_null([0] * 11),
    pl.col('kde_searchs').fill_null([0] * 11),
    pl.col('kde_buys').fill_null([0] * 11),
    pl.col('url_visit').fill_null([]),
    pl.col('sku_add').fill_null([]),
    pl.col('sku_rm').fill_null([]),
    pl.col('query_search').fill_null([0] * 64),
    pl.col('sku_buy').fill_null([]),
    pl.col('sku_add_cat').fill_null([]),
    pl.col('sku_rm_cat').fill_null([]),
    pl.col('sku_buy_cat').fill_null([])
)




ADD = 0
RM = 1
BUY = 2
SEARCH = 3
VISIT = 4

def process_user(user):
    actions = np.zeros((5,))
    actions[ADD] = user['adds']
    actions[RM] = user['rms']
    actions[BUY] = user['buys']
    actions[SEARCH] = user['searchs']
    actions[VISIT] = user['visits']
    
    cols = ['kde_visits', 'kde_adds', 'kde_rms', 'kde_searchs', 
            'kde_buys', 'query_search']
    cols_int = ['url_visit', 'sku_add', 'sku_rm', 'sku_buy',
                'sku_add_cat', 'sku_rm_cat', 'sku_buy_cat']
    return {'client': user['client_id'], 
            'actions': actions / np.sum(actions),
            } | {k: np.asarray(user[k], dtype=np.float32) for k in cols} | \
            {k: np.asarray(user[k], dtype=np.int64) for k in cols_int}





class UserBehaiviorDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return process_user(self.df[idx].to_dicts()[0])
        





us = UserBehaiviorDataset(df)





us[0]





def collate(data, kde_col = ['kde_visits', 'kde_adds', 'kde_rms', 'kde_searchs', 'kde_buys', 'query_search']):
    client = np.asarray([x['client'] for x in data], dtype=np.int64)
    actions = np.vstack([x['actions'] for x in data])
    
    kdes = {}
    for c in kde_col:
        kdes[c] = torch.from_numpy(np.vstack([x[c] for x in data])).float()

    url_visit = []
    sku_add = []
    sku_buy = []
    sku_rm = []
    sku_add_cat = []
    sku_buy_cat = []
    sku_rm_cat = []
    for i in range(len(data)):
        url_visit.append(torch.tensor(data[i]['url_visit']) + 1)
        sku_add.append(torch.tensor(data[i]['sku_add']) + 1)
        sku_rm.append(torch.tensor(data[i]['sku_rm']) + 1)
        sku_buy.append(torch.tensor(data[i]['sku_buy']) + 1)
        sku_add_cat.append(torch.tensor(data[i]['sku_add_cat']) + 1)
        sku_rm_cat.append(torch.tensor(data[i]['sku_rm_cat']) + 1)
        sku_buy_cat.append(torch.tensor(data[i]['sku_buy_cat']) + 1)

    #Create an array padding url_visit
    url_visit = pad_sequence(url_visit, batch_first=True, padding_value=0)
    sku_add = pad_sequence(sku_add, batch_first=True, padding_value=0)
    sku_buy = pad_sequence(sku_buy, batch_first=True, padding_value=0)
    sku_rm = pad_sequence(sku_rm, batch_first=True, padding_value=0)
    sku_add_cat = pad_sequence(sku_add_cat, batch_first=True, padding_value=0)
    sku_buy_cat = pad_sequence(sku_buy_cat, batch_first=True, padding_value=0)
    sku_rm_cat = pad_sequence(sku_rm_cat, batch_first=True, padding_value=0)

    return {'client': torch.from_numpy(client),
           'actions': torch.from_numpy(actions).float(),
           'url_visit': url_visit, 'sku_add': sku_add,
           'sku_rm': sku_rm, 'sku_buy': sku_buy, 
           'sku_add_cat': sku_add_cat, 'sku_rm_cat': sku_rm_cat, 
           'sku_buy_cat': sku_buy_cat} | kdes




def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

class ActionEncoder(nn.Module):

    def __init__(self, urls, skus, cats, user_emb=512, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.actions = nn.Linear(5, hidden_dim) 
        
        self.kde_add = nn.Linear(11, hidden_dim)
        self.kde_rms = nn.Linear(11, hidden_dim)
        self.kde_buys = nn.Linear(11, hidden_dim)
        self.kde_searchs = nn.Linear(11, hidden_dim)
        self.kde_visits = nn.Linear(11, hidden_dim)

        self.skus = nn.EmbeddingBag(skus + 1, hidden_dim, padding_idx=0)
        self.sku_add = nn.Linear(hidden_dim, hidden_dim)
        self.sku_rm = nn.Linear(hidden_dim, hidden_dim)
        self.sku_buy = nn.Linear(hidden_dim, hidden_dim)
        self.query_search = nn.Linear(64, hidden_dim)

        self.cats = nn.EmbeddingBag(cats + 1, hidden_dim, padding_idx=0)
        self.cat_add = nn.Linear(hidden_dim, hidden_dim)
        self.cat_rm = nn.Linear(hidden_dim, hidden_dim)
        self.cat_buy = nn.Linear(hidden_dim, hidden_dim)

        self.urls = nn.EmbeddingBag(urls + 1, hidden_dim, padding_idx=0)

        self.drop_mean = nn.Dropout(0.5)
        self.bn_mean = nn.BatchNorm1d(14 * hidden_dim)
        self.full_mean = nn.Linear(14 * hidden_dim, user_emb)
        self.drop_log_var = nn.Dropout(0.5)
        self.bn_log_var = nn.BatchNorm1d(14 * hidden_dim)
        self.full_log_var = nn.Linear(14 * hidden_dim, user_emb)
        pass

    def reparameterization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(var.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
    
    def forward(self, actions, sku_add, sku_rm, sku_buy, 
                sku_add_cat, sku_rm_cat, sku_buy_cat, 
                kde_add, kde_rms, kde_buys, kde_searchs, 
                kde_visits, query_search, url_visit):
        actions = self.actions(actions)
        kde_add = self.kde_add(kde_add)
        kde_rms = self.kde_rms(kde_rms)
        kde_buys = self.kde_buys(kde_buys)
        kde_searchs = self.kde_searchs(kde_searchs)
        kde_visits = self.kde_visits(kde_visits)
        sku_add = self.skus(sku_add)
        sku_rm = self.skus(sku_rm)
        sku_buy = self.skus(sku_buy)
        query_search = self.query_search(query_search)
        sku_add = self.sku_add(sku_add)
        sku_rm = self.sku_rm(sku_rm)
        sku_buy = self.sku_buy(sku_buy)

        sku_add_cat = self.cats(sku_add_cat)
        sku_rm_cat = self.cats(sku_rm_cat)
        sku_buy_cat = self.cats(sku_buy_cat)
        sku_add_cat = self.cat_add(sku_add_cat)
        sku_rm_cat = self.cat_rm(sku_rm_cat)
        sku_buy_cat = self.cat_buy(sku_buy_cat)
        url_visit = self.urls(url_visit)

        x_base = torch.cat([actions, kde_add, kde_rms, 
                            kde_buys, kde_searchs, kde_visits, 
                            sku_add, sku_rm, sku_buy, 
                            sku_add_cat, sku_rm_cat, sku_buy_cat, query_search, url_visit], dim=1)
        x = self.drop_mean(x_base)
        x = self.bn_mean(x)
        x = self.full_mean(x)
        if self.training:
            log_var = self.drop_log_var(x_base)
            log_var = self.bn_log_var(log_var)
            log_var = self.full_log_var(log_var)
            x = self.reparameterization(x, log_var)
        return x


class UserEncoder(nn.Module):

    def __init__(self, urls, skus, cats, user_emb=512, hidden_dim=128):
        super().__init__()
        self.model_1 = ActionEncoder(urls, skus, cats, user_emb, hidden_dim)
        self.model_2 = ActionEncoder(urls, skus, cats, user_emb, hidden_dim)
        pass

    def forward(self, actions, sku_add, sku_rm, sku_buy, 
                sku_add_cat, sku_rm_cat, sku_buy_cat, 
                kde_add, kde_rms, kde_buys, kde_searchs, kde_visits, query_search, url_visit):
        x1 = self.model_1(actions, sku_add, sku_rm, sku_buy,
                          sku_add_cat, sku_rm_cat, sku_buy_cat, kde_add, 
                          kde_rms, kde_buys, kde_searchs, kde_visits, query_search, url_visit)
        x2 = self.model_2(actions, sku_add, sku_rm, sku_buy,
                          sku_add_cat, sku_rm_cat, sku_buy_cat, kde_add, 
                          kde_rms, kde_buys, kde_searchs, kde_visits, query_search, url_visit)
        return x1, x2
    
print('Creating model')
model = UserEncoder(len(map_urls_id), len(map_prod_id), len(map_cats_id))
model = torch.optim.swa_utils.AveragedModel(model, 
                                                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
model.load_state_dict(torch.load('emb-8ShortVAECATURLFastNAdamWEMA_best.pth'))

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu' 
batch_size = 1000

model = model.to(device)
model.eval()

with torch.no_grad():
    dl = DataLoader(us, batch_size=batch_size, collate_fn=collate, shuffle=False)
    clients = []
    encode_1 = []
    encode_2 = []
    for x in tqdm(dl):
        
        client = x['client'].to(device)
        actions = x['actions'].to(device)
        sku_add = x['sku_add'].to(device)
        sku_rm = x['sku_rm'].to(device)
        sku_buy = x['sku_buy'].to(device)
        sku_add_cat = x['sku_add_cat'].to(device)
        sku_rm_cat = x['sku_rm_cat'].to(device)
        sku_buy_cat = x['sku_buy_cat'].to(device)
        kde_visits = x['kde_visits'].to(device)
        kde_adds = x['kde_adds'].to(device)
        kde_rms = x['kde_rms'].to(device)
        kde_searchs = x['kde_searchs'].to(device)
        kde_buys = x['kde_buys'].to(device)
        query_search = x['query_search'].to(device)
        url_visit = x['url_visit'].to(device)

        y1, y2 = model(actions, sku_add, sku_rm, sku_buy,
                       sku_add_cat, sku_rm_cat, sku_buy_cat,
                       kde_adds, kde_rms, kde_buys, kde_searchs, 
                       kde_visits, query_search, url_visit)

        clients.extend([client_ids[x] for x in client.cpu().numpy().tolist()])
        encode_1.append(y1.cpu().numpy())
        encode_2.append(y2.cpu().numpy())


print('Cleaning...')
del dl
del us
del df
del model
import gc
print(gc.collect())

print('Saving...')

encode_1 = np.concatenate(encode_1, axis=0)
clients = np.asarray(clients, dtype=np.int64)
encode_2 = np.concatenate(encode_2, axis=0)



print('Saving combined...')
encode = np.concatenate([encode_1, encode_2], axis=1)
print(encode.shape)
os.makedirs('emb-8ShortVAECATURLFullNAEMAAVG', exist_ok=True)
np.save('emb-8ShortVAECATURLFullNAEMAAVG/client_ids.npy', clients)
np.save('emb-8ShortVAECATURLFullNAEMAAVG/embeddings.npy', encode.astype(np.float16))