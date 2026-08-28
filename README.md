<div align="center">

# IMDB Sentiment Analysis

**Three deep-learning architectures compared on IMDB movie-review sentiment classification, with a live Streamlit demo.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io/)

[English](README.md) · [Türkçe](README.tr.md)

</div>

---

## Overview

A custom **Multi-Kernel TextCNN** is benchmarked against two literature baselines — **LSTM** and
**Bidirectional LSTM** — on the IMDB binary sentiment dataset (25,000 test reviews). All three
models are trained from scratch; no transfer learning or pre-trained embeddings are used.

A Streamlit app lets you run any model on your own text, compare all three side by side, and
inspect the training curves and confusion matrices.

## Results

Evaluated on the full 25,000-review IMDB test split.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Multi-Kernel TextCNN** (custom) | **0.8710** | 0.8725 | 0.8690 | **0.8707** |
| Bidirectional LSTM | 0.8484 | 0.8434 | 0.8558 | 0.8495 |
| LSTM | 0.8387 | 0.8186 | 0.8703 | 0.8437 |

The custom TextCNN wins on every metric. Convolutional n-gram features turn out to be both
stronger and far cheaper to train than the recurrent baselines on this task.

> [!NOTE]
> All three models overfit: TextCNN reaches 97.7% training accuracy against 87.0% validation.
> Validation loss bottoms out around epoch 2 and rises afterwards. Early stopping or stronger
> regularisation would be the first thing to add.

## Architectures

Shared across all three models:

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Max sequence length | 250 |
| Embedding dimension | 128 |
| Batch size | 64 |
| Optimizer | Adam (lr = 0.001) |
| Seed | 42 |

| Model | Distinguishing setup | Epochs |
|---|---|---:|
| Multi-Kernel TextCNN | Parallel convolutions over several kernel widths | 5 |
| LSTM | 128 LSTM units | 10 |
| Bidirectional LSTM | 64 units per direction (128 output dim) | 5 |

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

> [!NOTE]
> The first prediction takes 5–10 s while the models load; subsequent ones are under a second.

### Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**, select this repo and the `main` branch.
2. **Main file path:** `app/streamlit_app.py`
3. **Advanced settings → Python version:** 3.11
4. **Deploy.** The first build takes 3–5 minutes.

## App Tabs

| Tab | What it does |
|---|---|
| **Single Model** | Pick one model, enter a review, get a prediction. |
| **Compare All Three** | Send the same review to all models, see scores side by side. |
| **Model Analysis** | Accuracy / precision / recall / F1, training curves, confusion matrices. |

## Project Structure

```
.
├── app/
│   ├── streamlit_app.py     Main UI (3 tabs)
│   ├── model_loader.py      Cached loading of all three models
│   └── preprocess.py        IMDB tokenisation + padding
├── model1/                  Multi-Kernel TextCNN — weights, notebook, plots, results JSON
├── model2/                  LSTM — same layout
├── model3/                  Bidirectional LSTM — same layout
├── docs/
│   ├── project-brief.pdf
│   └── presentation-outline.md
├── requirements.txt
├── runtime.txt              python-3.11
└── .streamlit/config.toml   Theme
```

Each `modelN/` folder holds its trained `.keras` weights, the training notebook, the training
curves, a confusion matrix and a `results_model_N.json` with the exact metrics quoted above.

## Limitations

- Trained on **English** reviews only — accuracy drops on any other language.
- No transfer learning; all weights are learned from scratch.
- Vocabulary is capped at the 10,000 most frequent tokens, so rare words are mapped to `<UNK>`.

## Course

**BIM-430 Deep Learning** term project.

## License

Released under the [MIT License](LICENSE).
