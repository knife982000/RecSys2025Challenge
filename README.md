# Dargk Team Submission - RecSys 2025 Challenge

Welcome to the official repository of the **Dargk** team submission for the [RecSys 2025 Challenge](https://recsys.synerise.com/). This repository contains the full pipeline and implementation of our model, **BEHAV-E**: _Behavioral Embedding via Hybrid Action Variational Encoder_.

Our approach is centered on modeling user behavior through a hybrid representation combining semantic, categorical, and temporal information, and training via a contrastive learning framework.

---

## 🧠 Model Overview

### BEHAV-E: Behavioral Embedding via Hybrid Action Variational Encoder

BEHAV-E is a representation learning model designed to encode complex and multi-faceted user behavior. The model processes various user actions such as:

- Product **buys**
- Product **add-to-cart**
- Product **remove-from-cart**
- **URL** visits
- **Search** queries

Key modeling elements include:

- **Kernel Density Estimation (KDE)** to capture temporal distribution of actions.
- **LSTM-based Autoencoder** to embed user search queries.
- **Shared Embedding Bags** with linear transformations to efficiently represent items, categories, and URLs.
- **Variational Encoder** to output user behavior embeddings using reparameterization trick.
- **Contrastive Learning (InfoNCE loss)** with dual models to learn discriminative embeddings.

During inference, we concatenate the mean embeddings from two BEHAV-E models as an ensemble strategy.

---

## 📁 Repository Structure

* `EmbeddingAnalysis`: contains the code for reproduce the analysis as well as more details than the ones presented on the paper.
* `data/`: Folder where the decompressed dataset should be placed
* `01-TextEncodeTrain.py`: Trains the LSTM autoencoder for text embeddings
* `02-TextEncodeProcess.py`: Encodes the search text using the trained autoencoder
* `03-Polars_DS_enc_search.py`: Preprocesses data: one row per client format
* `04-Emb-8ShortVAECATURLFastNAEMA.py`: Trains BEHAV-E using EMA (Exponential Moving Average)
* `04-Emb-8ShortVAECATURLFastNAOneCycleLarge.py`: Trains BEHAV-E using One Cycle LR schedule
* `05-Emb-8ShortVAECATURLFastNAEMA_gen.py`: Generates embeddings using EMA-trained model
* `05-Emb-8ShortVAECATURLFastNAOneCycleLarge_gen.py`: Generates embeddings using OneCycle-trained model
* `06-Merger.py`: Concatenates embeddings for final submission
* `environment.yml`: Conda environment definition for Windows
* `Dockerfile`: Dockerfile to build a docker container for out model.
* `README.md`: This file


**Execution Note:** Files should be executed sequentially from `01` to `06`. Files with the same prefix number (e.g., both `04-*`) can be run in any order.

---

## ⚙️ Setup 

There are two ways to setup teh environment. The first one is in the local environment through conda with the provided `environment.yml` file. This has been tested only on a Windows environment. The second way is using a docker container with the provided `Dockerfile`.

Notice that this project only depends on the data, no pretrained model is required to run the pipeline end-to-end. However, we provide our trained models to reproduce the embeddings submitted to the challenge. The link is provided below.

1. **Clone the repository:**
```bash
git clone https://github.com/<your-org>/recsys2025-dargk.git
cd recsys2025-dargk
```

### 💻 Local


1. **Set up the conda environment:**

```bash
conda env create -f environment.yml
conda activate behav-e
```

### 🐳 Docker (Recommended)
Follow these instructions to build and run the RecSys2025 Docker container with GPU support.

1. **Build the Docker Image**
```bash
docker build -t recsys2025 .
```

2. **Create the Docker Container**
   1. Windows
    ```cmd 
    docker container create -i -t -v "%CD%":/recsys2025 --gpus=all --name recsys2025 recsys2025
    ```
   2. Linux/macOS
    ```bash 
    docker container create -i -t -v "$PWD":/recsys2025 --gpus=all --name recsys2025 recsys2025
    ```
3. **Start and Attach to the Container**
```bash
docker container start --attach -i recsys2025
```
## 💾 Preparing the dataData
To run the system is necesary to download [the dataset](https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz), which is compressed in a file calles `ubc_data.tar.gz`. This file must be extracted into the `data/` directory.

A detailed description of the dataset can be found at the challenge [Web site](https://recsys.synerise.com/summary#dataset). It containes six parquet files:

* `product_buy.parquet`
    * client_id (int64): Numeric ID of the client (user).
    * timestamp (object): Date and time of the event in the format YYYY-MM-DD HH:mm:ss.
    * sku (int64): Numeric ID of the item.

* `add_to_cart.parquet`
    * client_id (int64): Numeric ID of the client (user).
    * timestamp (object): Date and time of the event in the format YYYY-MM-DD HH:mm:ss.
    * sku (int64): Numeric ID of the item.

* `remove_from_cart.parquet`
    * client_id (int64): Numeric ID of the client (user).
    * timestamp (object): Date and time of the event in the format YYYY-MM-DD HH:mm:ss.
    * sku (int64): Numeric ID of the item.

* `product_properties.parquet`
    * sku (int64): Numeric ID of the item.
    * category (int64): Numeric ID of the item category.
    * price (int64): Numeric ID of the item's price bucket.
    * embedding (object): A textual embedding of a product name, compressed using the product quantization method.

* `page_visit.parquet`
    * client_id (int64): Numeric ID of the client.
    * timestamp (object): Date and time of the event in the format YYYY-MM-DD HH:mm
    * url (int64): Numeric ID of a visited URL. The explicit information about what (e.g., which item) is presented on a particular page is not provided.

* `search_query.parquet`
    * client_id (int64): Numeric ID of the client.
    * timestamp (object): Date and time of the event in the format YYYY-MM-DD HH:mm:ss.
    * query (object): The textual embedding of the search query, compressed using the product quantization method.
‍
The dataset also containes two more directories related to the task.
* `input` directory: This directory stores a NumPy file containing a subset of 1,000,000 client_ids for which Universal Behavioral Profiles should be generated:
    * relevant_clients.npy

* `target` directory: This directory stores the labels for propensity tasks. For each propensity task, target category names are stored in NumPy files:

  * `propensity_category.npy`: Contains a subset of 100 categories for which the model is asked to provide predictions
  * `popularity_propensity_category.npy`: Contains popularity scores for categories from the `propensity_category.npy` file. Scores are used to compute the Novelty measure.
  * `propensity_sku.npy`: Contains a subset of 100 products for which the model is asked to provide predictions
  * `popularity_propensity_sku.npy`: Contains popularity scores for products from the propensity_sku.npy file. These scores are used to compute the Novelty measure.
  * `active_clients.npy`: Contains a subset of relevant clients with at least one product_buy event in history (data available for the participants). Active clients are used to compute churn target.

---

## 🚀 Pipeline Execution

Follow the steps below to reproduce the full embedding generation and submission process.

1. **Train the LSTM Autoencoder for Search Query Embeddings**
```bash
python 01-TextEncodeTrain.py
```

2. **Encode the Search Texts**
```bash
python 02-TextEncodeProcess.py
```

3. **Preprocess the Dataset to Generate One Row per User**

```bash
python 03-Polars_DS_enc_search.py
```

4. **Train the BEHAV-E Models**

* **Exponential Moving Average (EMA) Variant:**
```bash
python 04-Emb-8ShortVAECATURLFastNAEMA.py
```

* **One Cycle Learning Rate Variant:**

```bash
python 04-Emb-8ShortVAECATURLFastNAOneCycleLarge.py
```

5. **Generate User Embeddings from Trained Models**

* **From EMA-trained model:**

```bash
python 05-Emb-8ShortVAECATURLFastNAEMA_gen.py
```

* **From OneCycle-trained model:*

```bash
python 05-Emb-8ShortVAECATURLFastNAOneCycleLarge_gen.py
``` 

6. **Merge the Generated Embeddings for Final Submission**

```bash
python 06-Merger.py
```

⚠️ **Note:** All scripts with the same prefix number (e.g., 04-*, 05-*) can be run in any order.

## 🧪 Inference Details

At inference time:

- BEHAV-E uses the **mean** of the latent distribution, i.e., $$\mu = \mathbb{E}[z]$$, instead of sampling from $$\mathcal{N}(\mu, \sigma^2)$$.
- **Dropout** is disabled to ensure deterministic outputs.
- The final user embedding is the **concatenation of the embeddings** produced by both trained BEHAV-E models (EMA and OneCycle variants). Although both models are trained on the same data, this concatenation acts as a lightweight ensemble and improves robustness.

---

## Submission

The pretrained models used for generate our submission can be downloaded from [an external repository](https://drive.google.com/file/d/1MqaLrqhDa46fUmu89RyVt9iP95B0Yg54/view?usp=sharing).

---

## 🙌 Acknowledgements

We would like to thank the organizers of the [RecSys 2025 Challenge](https://recsys.synerise.com/) for providing a valuable dataset and a well-designed competition platform.

---

## 📫 Contacts

For inquiries, collaboration, or questions regarding this submission, please contact the **Dargk Team** at:

📧 [Antonela Tommasel](https://tommantonela.github.io) (antonela.tommasel@isistan.unicen.edu.ar)

📧 [Juan Manuel Rodriguez](https://sites.google.com/site/rodriguezjuanmanuel/home) (jmro@cs.aau.dk)
