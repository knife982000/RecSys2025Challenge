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

* EmbeddingAnalysis: contains the code for reproduce the analysis as well as more details than the ones presented on the paper.
* data/ # Folder where the decompressed dataset should be placed
* 01-TextEncodeTrain.py # Trains the LSTM autoencoder for text embeddings
* 02-TextEncodeProcess.py # Encodes the search text using the trained autoencoder
* 03-Polars_DS_enc_search.py # Preprocesses data: one row per client format
* 04-Emb-8ShortVAECATURLFastNAEMA.py # Trains BEHAV-E using EMA (Exponential Moving Average)
* 04-Emb-8ShortVAECATURLFastNAOneCycleLarge.py # Trains BEHAV-E using One Cycle LR schedule
* 05-Emb-8ShortVAECATURLFastNAEMA_gen.py # Generates embeddings using EMA-trained model
* 05-Emb-8ShortVAECATURLFastNAOneCycleLarge_gen.py # Generates embeddings using OneCycle-trained model
* 06-Merger.py # Concatenates embeddings for final submission
* environment.yml # Conda environment definition
* README.md # This file


**Execution Note:** Files should be executed sequentially from `01` to `06`. Files with the same prefix number (e.g., both `04-*`) can be run in any order.

---

## ⚙️ Setup

1. **Clone the repository:**
```bash
git clone https://github.com/<your-org>/recsys2025-dargk.git
cd recsys2025-dargk
```

2. **Set up the conda environment:**

```bash
conda env create -f environment.yml
conda activate behav-e
```

3. **Prepare the dataset:**

* Download and extract the dataset into the data/ directory.

4. **Pretrained models**

* Pretrained models can be downloaded from [an external repository](https://drive.google.com/file/d/1MqaLrqhDa46fUmu89RyVt9iP95B0Yg54/view?usp=sharing).
## ⚙️ Conda Environment

Create the required conda environment:

```bash
conda env create -f environment.yml
conda activate behav-e
```
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

## 🙌 Acknowledgements

We would like to thank the organizers of the [RecSys 2025 Challenge](https://recsys.synerise.com/) for providing a valuable dataset and a well-designed competition platform.

---

## 📫 Contact

For inquiries, collaboration, or questions regarding this submission, please contact the **Dargk Team** at:

📧 [Antonela Tommasel](https://tommantonela.github.io) (antonela.tommasel@isistan.unicen.edu.ar)

📧 [Juan Manuel Rodriguez](https://sites.google.com/site/rodriguezjuanmanuel/home) (jmro@cs.aau.dk)