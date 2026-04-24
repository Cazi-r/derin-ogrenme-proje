"""3 Keras modelini tembel (lazy) + cache'li olarak yükler."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    short: str
    folder: str
    weights: str
    results_json: str
    training_curve: str
    confusion_matrix: str
    architecture: str | None
    readme: str


MODELS: dict[str, ModelSpec] = {
    "textcnn": ModelSpec(
        key="textcnn",
        display_name="Model 1 — Multi-Kernel TextCNN",
        short="TextCNN",
        folder="model1",
        weights="model_1_custom_textcnn.keras",
        results_json="results_model_1.json",
        training_curve="model1_training_curves.png",
        confusion_matrix="model1_confusion_matrix.png",
        architecture=None,
        readme="README.md",
    ),
    "lstm": ModelSpec(
        key="lstm",
        display_name="Model 2 — LSTM",
        short="LSTM",
        folder="model2",
        weights="model_2_lstm.keras",
        results_json="results_model_2.json",
        training_curve="model2_training_curves.png",
        confusion_matrix="model2_confusion_matrix.png",
        architecture="model2_architecture.png",
        readme="README.md",
    ),
    "bilstm": ModelSpec(
        key="bilstm",
        display_name="Model 3 — BiLSTM",
        short="BiLSTM",
        folder="model3",
        weights="model_3_bilstm.keras",
        results_json="results_model_3.json",
        training_curve="model3_training_curves.png",
        confusion_matrix="model3_confusion_matrix.png",
        architecture="model3_architecture.png",
        readme="README.md",
    ),
}


def asset_path(spec: ModelSpec, filename: str) -> Path:
    return ROOT / spec.folder / filename


@st.cache_resource(show_spinner="Model yükleniyor...")
def load_model(key: str) -> tf.keras.Model:
    spec = MODELS[key]
    return tf.keras.models.load_model(asset_path(spec, spec.weights), compile=False)


@st.cache_data(show_spinner=False)
def load_results(key: str) -> dict:
    spec = MODELS[key]
    return json.loads(asset_path(spec, spec.results_json).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_readme(key: str) -> str:
    spec = MODELS[key]
    return asset_path(spec, spec.readme).read_text(encoding="utf-8")
