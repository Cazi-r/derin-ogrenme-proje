"""IMDB yorumlarını modele uygun sayısal diziye çevirir."""
from __future__ import annotations

import re

import numpy as np
import streamlit as st
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 250
INDEX_OFFSET = 3
PAD_ID = 0
START_ID = 1
UNK_ID = 2


@st.cache_data(show_spinner=False)
def load_word_index() -> dict[str, int]:
    raw = imdb.get_word_index()
    return {w: i + INDEX_OFFSET for w, i in raw.items()}


_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def encode(text: str) -> np.ndarray:
    word_index = load_word_index()
    ids = [START_ID]
    for tok in _tokenize(text):
        idx = word_index.get(tok, UNK_ID)
        ids.append(idx if idx < VOCAB_SIZE else UNK_ID)
    return pad_sequences([ids], maxlen=MAX_LEN, padding="pre", truncating="pre")
