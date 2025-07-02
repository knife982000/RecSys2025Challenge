import numpy as np
import os

paths = ['emb-8ShortVAECATURLFullNAOCLarAVG/', 'emb-8ShortVAECATURLFullNAEMAAVG/']

data = []
for p in paths:
    clients = np.load(p + 'client_ids.npy')
    encode = np.load(p + 'embeddings.npy')
    data.append((clients, encode))

for c, _ in data:
    assert np.all(c == clients)

data = np.concatenate([x[1] for x in data], axis=1)
print(data.shape)
print(data.dtype)

os.makedirs('EMA_OneCycleLarge', exist_ok=True)

np.save('EMA_OneCycleLarge/client_ids.npy', clients)
np.save('EMA_OneCycleLarge/embeddings.npy', data)