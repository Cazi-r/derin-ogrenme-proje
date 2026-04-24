"""IMDB Sentiment Analysis — 3 Model Canlı Demo."""
from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from model_loader import MODELS, ModelSpec, asset_path, load_model, load_readme, load_results
from preprocess import encode, load_test_samples

st.set_page_config(
    page_title="IMDB Duygu Analizi — 3 Model",
    page_icon="🎬",
    layout="wide",
)

EXAMPLES = {
    "Pozitif örnek": "This movie was absolutely fantastic. The acting was superb and the story kept me engaged from start to finish. Highly recommended!",
    "Negatif örnek": "Worst film ever. Complete waste of time, the plot made no sense and the acting was painfully bad.",
}

def pick_random_review(exclude: str | None = None) -> tuple[str, int]:
    """IMDB test setinden rastgele (text, label) döndürür. label: 0=Negatif, 1=Pozitif."""
    samples = load_test_samples()
    for _ in range(10):
        text, label = random.choice(samples)
        if text != exclude:
            return text, label
    return random.choice(samples)


def predict(model_key: str, text: str) -> float:
    x = encode(text)
    model = load_model(model_key)
    prob = float(model.predict(x, verbose=0)[0][0])
    return prob


def label_for(prob: float) -> tuple[str, str]:
    if prob >= 0.5:
        return "Pozitif 😊", "positive"
    return "Negatif 😞", "negative"


def render_prediction_card(spec: ModelSpec, prob: float) -> None:
    label, _ = label_for(prob)
    confidence = prob if prob >= 0.5 else 1 - prob
    st.markdown(f"### {spec.short}")
    st.markdown(f"**Tahmin:** {label}")
    st.progress(confidence, text=f"Güven: %{confidence * 100:.1f}")
    st.caption(f"Pozitif olasılığı: {prob:.4f}")


def sidebar_accuracy_badges() -> None:
    st.sidebar.markdown("### Model Doğrulukları (Test)")
    for spec in MODELS.values():
        acc = load_results(spec.key)["accuracy"]
        st.sidebar.markdown(f"- **{spec.short}** — %{acc * 100:.2f}")


# ---------- Header ----------
st.title("🎬 IMDB Duygu Analizi")
st.caption(
    "3 farklı derin öğrenme modeli (TextCNN · LSTM · BiLSTM) ile İngilizce film "
    "yorumlarından duygu tahmini. BIM 430 Derin Öğrenme Projesi."
)

with st.sidebar:
    st.markdown("## 🎛️ Kontrol Paneli")
    sidebar_accuracy_badges()
    st.markdown("---")
    st.info("Modeller IMDB (İngilizce) veri setiyle eğitildi. En doğru sonuç için İngilizce yorum girin.")

tab_single, tab_compare, tab_analysis = st.tabs(
    ["🔎 Tek Model", "⚖️ 3 Modeli Karşılaştır", "📊 Model Analizi"]
)

# ---------- Tab 1: Single ----------
with tab_single:
    col_input, col_model = st.columns([3, 1])
    with col_model:
        choice = st.radio(
            "Model seç",
            options=list(MODELS.keys()),
            format_func=lambda k: MODELS[k].short,
            key="single_model",
        )
    with col_input:
        st.markdown("#### Film yorumunu girin (İngilizce)")
        ex_cols = st.columns(len(EXAMPLES) + 1)
        for i, (name, sample) in enumerate(EXAMPLES.items()):
            if ex_cols[i].button(name, key=f"ex_single_{i}"):
                st.session_state["single_text"] = sample
                st.session_state.pop("single_truth", None)
        if ex_cols[-1].button("🎲 IMDB'den rastgele", key="rand_single"):
            text_rand, label_rand = pick_random_review(
                exclude=st.session_state.get("single_text")
            )
            st.session_state["single_text"] = text_rand
            st.session_state["single_truth"] = label_rand
        if "single_truth" in st.session_state:
            truth_txt = "Pozitif" if st.session_state["single_truth"] == 1 else "Negatif"
            st.caption(f"📎 Gerçek etiket (IMDB test seti): **{truth_txt}**")
        text = st.text_area(
            "Yorum",
            key="single_text",
            height=180,
            placeholder="Write an English movie review here...",
            label_visibility="collapsed",
        )
        run = st.button("🚀 Analiz Et", type="primary", use_container_width=True)

    if run:
        if not text or not text.strip():
            st.warning("Lütfen önce bir yorum girin.")
        else:
            prob = predict(choice, text)
            st.markdown("---")
            spec = MODELS[choice]
            label, kind = label_for(prob)
            confidence = prob if prob >= 0.5 else 1 - prob
            color = "#10b981" if kind == "positive" else "#ef4444"
            truth = st.session_state.get("single_truth")
            if truth is not None:
                correct = (truth == 1 and kind == "positive") or (truth == 0 and kind == "negative")
                badge = "✅ Doğru tahmin" if correct else "❌ Yanlış tahmin"
                st.markdown(f"**{badge}** — gerçek: {'Pozitif' if truth == 1 else 'Negatif'}")
            st.markdown(
                f"""
                <div style="padding:1.25rem;border-radius:12px;border:1px solid {color};background:{color}15;">
                    <div style="font-size:0.9rem;opacity:0.75">{spec.display_name}</div>
                    <div style="font-size:2rem;font-weight:700;color:{color}">{label}</div>
                    <div style="font-size:1rem;margin-top:0.25rem">Güven: <b>%{confidence * 100:.1f}</b> &nbsp;·&nbsp; Pozitif olasılığı: {prob:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------- Tab 2: Compare ----------
with tab_compare:
    st.markdown("#### Aynı yorumu 3 modele birden gönderin")
    ex_cmp_cols = st.columns(len(EXAMPLES) + 1)
    for i, (name, sample) in enumerate(EXAMPLES.items()):
        if ex_cmp_cols[i].button(name, key=f"ex_cmp_{i}"):
            st.session_state["compare_text"] = sample
            st.session_state.pop("compare_truth", None)
    if ex_cmp_cols[-1].button("🎲 IMDB'den rastgele", key="rand_cmp"):
        text_rand, label_rand = pick_random_review(
            exclude=st.session_state.get("compare_text")
        )
        st.session_state["compare_text"] = text_rand
        st.session_state["compare_truth"] = label_rand
    if "compare_truth" in st.session_state:
        truth_txt = "Pozitif" if st.session_state["compare_truth"] == 1 else "Negatif"
        st.caption(f"📎 Gerçek etiket (IMDB test seti): **{truth_txt}**")
    cmp_text = st.text_area(
        "Yorum",
        key="compare_text",
        height=160,
        placeholder="Write an English movie review here...",
        label_visibility="collapsed",
    )
    run_all = st.button("⚖️ Hepsini Çalıştır", type="primary", use_container_width=True)

    if run_all:
        if not cmp_text or not cmp_text.strip():
            st.warning("Lütfen önce bir yorum girin.")
        else:
            results = {k: predict(k, cmp_text) for k in MODELS}
            cols = st.columns(len(MODELS))
            for col, (k, prob) in zip(cols, results.items()):
                with col:
                    render_prediction_card(MODELS[k], prob)

            st.markdown("---")
            df = pd.DataFrame(
                {
                    "Model": [MODELS[k].short for k in results],
                    "Pozitif olasılığı": [results[k] for k in results],
                    "Negatif olasılığı": [1 - results[k] for k in results],
                }
            ).set_index("Model")
            st.bar_chart(df)

            labels = [label_for(p)[1] for p in results.values()]
            if len(set(labels)) == 1:
                kind = labels[0]
                st.success(f"✅ 3 model de aynı fikirde: **{label_for(results[next(iter(results))])[0]}**")
            else:
                pos = sum(1 for k in labels if k == "positive")
                st.warning(f"⚠️ Modeller ayrıştı — {pos} Pozitif, {3 - pos} Negatif")

            truth_cmp = st.session_state.get("compare_truth")
            if truth_cmp is not None:
                truth_kind = "positive" if truth_cmp == 1 else "negative"
                correct_count = sum(1 for lbl in labels if lbl == truth_kind)
                st.info(
                    f"🎯 Gerçek etiket: **{'Pozitif' if truth_cmp == 1 else 'Negatif'}** — "
                    f"{correct_count}/3 model doğru bildi."
                )

# ---------- Tab 3: Analysis ----------
with tab_analysis:
    st.markdown("#### Eğitim metrikleri ve grafikler")
    pick = st.radio(
        "Model",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k].display_name,
        horizontal=True,
    )
    spec = MODELS[pick]
    res = load_results(pick)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{res['accuracy']:.4f}")
    m2.metric("Precision", f"{res['precision']:.4f}")
    m3.metric("Recall", f"{res['recall']:.4f}")
    m4.metric("F1-Score", f"{res['f1_score']:.4f}")

    st.markdown("---")
    img_cols = st.columns(2)
    with img_cols[0]:
        st.markdown("**Eğitim Eğrileri**")
        st.image(str(asset_path(spec, spec.training_curve)), use_column_width=True)
    with img_cols[1]:
        st.markdown("**Confusion Matrix**")
        st.image(str(asset_path(spec, spec.confusion_matrix)), use_column_width=True)

    if spec.architecture:
        st.markdown("**Model Mimarisi**")
        st.image(str(asset_path(spec, spec.architecture)), use_column_width=True)

    with st.expander("📄 Model README"):
        st.markdown(load_readme(pick))

st.markdown("---")
st.caption("BIM 430 — Derin Öğrenme Projesi · IMDB Sentiment Analysis · Streamlit Cloud")
