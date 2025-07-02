#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import polars as pl
import numpy as np
import os
from datetime import datetime
from tqdm.auto import tqdm
from pymongo import MongoClient
import pymongo
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def to_int_list(x):
    x = x[1:-1].split(' ')
    x = [int(i) for i in x if i != '']
    return x


# In[ ]:


data_dir = 'data'
data = {}
for f_name in os.listdir(data_dir):
    if f_name.endswith('.parquet'):
        name = f_name.split('.')[0]
        print(f'Loading {name}...')
        data[name] = pl.read_parquet(f'{data_dir}{os.sep}{f_name}')
        data[name]= data[name].unique()
        if name != 'product_properties':
            # data[name] = data[name].with_columns(pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S"))
            data[name] = data[name].sort(['client_id', 'timestamp'])
        if name == 'search_query':
            data[name] = data[name].with_columns(pl.col('query').map_elements(to_int_list, return_dtype=pl.List(pl.Int64)))
        if name == 'product_properties':
            data[name] = data[name].with_columns(pl.col('name').map_elements(to_int_list, return_dtype=pl.List(pl.Int64)))


# In[ ]:


data['search_query'] = pl.read_parquet(f'search_query.parquet')  # .with_columns(pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S"))
#data['product_properties'] = pl.read_parquet(f'product_properties.parquet').select(['sku', 'name'])


# In[ ]:


events = ['add_to_cart', 'page_visit', 'product_buy', 'remove_from_cart', 'search_query']
product = 'product_properties'


# In[ ]:


## Timestamp %
min_date = min([data[x]['timestamp'].min() for x in events]).timestamp()
max_date = max([data[x]['timestamp'].max() for x in events]).timestamp()
diff = max_date - min_date


# In[ ]:


data['add_to_cart'] = data['add_to_cart'].with_columns(
    pl.col('timestamp').map_elements(lambda x: (x.timestamp() - min_date) / diff,
                                    return_dtype=pl.Float64).cast(pl.Float32).alias('ts_proportional_add'))

data['page_visit'] = data['page_visit'].with_columns(
    pl.col('timestamp').map_elements(lambda x: (x.timestamp() - min_date) / diff,
                                    return_dtype=pl.Float64).cast(pl.Float32).alias('ts_proportional_visit'))

data['product_buy'] = data['product_buy'].with_columns(
    pl.col('timestamp').map_elements(lambda x: (x.timestamp() - min_date) / diff,
                                    return_dtype=pl.Float64).cast(pl.Float32).alias('ts_proportional_buy'))

data['remove_from_cart'] = data['remove_from_cart'].with_columns(
    pl.col('timestamp').map_elements(lambda x: (x.timestamp() - min_date) / diff,
                                    return_dtype=pl.Float64).cast(pl.Float32).alias('ts_proportional_rm'))

data['search_query'] = data['search_query'].with_columns(
    pl.col('timestamp').map_elements(lambda x: (x.timestamp() - min_date) / diff,
                                    return_dtype=pl.Float64).cast(pl.Float32).alias('ts_proportional_search'))


# In[ ]:


def group_by(df, column='client_id'):
    return df.group_by(column).agg([pl.col(x) for x in df.columns if x != column])


# In[ ]:


def merge_keys(df, suffix):
    df = df.with_columns(pl.struct([
        pl.col('client_id'), 
        pl.col(f'client_id{suffix}')]).
                         map_elements(lambda x: x['client_id'] if x['client_id'] is not None else x[f'client_id{suffix}'], 
                                      return_dtype=pl.Int64).
                         alias('client_id'))
    return df.drop(f'client_id{suffix}')
    

def merge(data):
    df = data['add_to_cart'].rename({'timestamp':'timestamp_add', 'sku':'sku_add'}) 
    df = group_by(df)
    df = df.join(group_by(data['page_visit'].rename({'timestamp':'timestamp_visit', 'url':'url_visit'})), how='full', on='client_id', suffix='_visit')
    df = merge_keys(df, suffix='_visit')
    df = df.join(group_by(data['product_buy'].rename({'timestamp':'timestamp_buy', 'sku':'sku_buy'})), how='full', on='client_id', suffix='_buy')
    df = merge_keys(df, suffix='_buy') 
    df = df.join(group_by(data['remove_from_cart'].rename({'timestamp':'timestamp_rm', 'sku':'sku_rm'})), how='full', on='client_id', suffix='_rm')
    df = merge_keys(df, suffix='_rm')
    df = df.join(group_by(data['search_query'].rename({'timestamp':'timestamp_search', 'query':'query_search'})), how='full', on='client_id', suffix='_search')
    df = merge_keys(df, suffix='_search') 
    return df.sort('client_id')

def merge_no_visit(data):
    df = data['add_to_cart'].rename({'timestamp':'timestamp_add', 'sku':'sku_add'}) 
    df = group_by(df)
    df = df.join(group_by(data['product_buy'].rename({'timestamp':'timestamp_buy', 'sku':'sku_buy'})), how='full', on='client_id', suffix='_buy')
    df = merge_keys(df, suffix='_buy') 
    df = df.join(group_by(data['remove_from_cart'].rename({'timestamp':'timestamp_rm', 'sku':'sku_rm'})), how='full', on='client_id', suffix='_rm')
    df = merge_keys(df, suffix='_rm')
    df = df.join(group_by(data['search_query'].rename({'timestamp':'timestamp_search', 'query':'query_search'})), how='full', on='client_id', suffix='_search')
    df = merge_keys(df, suffix='_search') 
    return df.sort('client_id')


# In[ ]:


df = merge(data)


# In[ ]:


df


# In[ ]:


len(df)


# In[ ]:


df.write_parquet('full_enc_search.parquet')


# In[ ]:


del df


# In[ ]:


df = merge_no_visit(data)


# In[ ]:


df


# In[ ]:


len(df)


# In[ ]:


df.write_parquet('limited_enc_search.parquet')


# In[ ]:




