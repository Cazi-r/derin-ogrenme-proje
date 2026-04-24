# IMDB Duygu Analizi — 3 Model Karşılaştırmalı Demo

BIM 430 Derin Öğrenme Projesi. IMDB film yorumları üzerinde **Multi-Kernel TextCNN**, **LSTM** ve **BiLSTM** modelleriyle duygu analizi (sentiment analysis). Streamlit Community Cloud üzerinde canlı demo.

## 📦 Proje Yapısı

```
derin-ogrenme-proje/
├── app/
│   ├── streamlit_app.py     # Ana UI (3 sekme)
│   ├── model_loader.py      # 3 modeli cache'li yükle
│   └── preprocess.py        # IMDB tokenize + pad
├── model1/ model2/ model3/  # Eğitilmiş .keras dosyaları + grafikler + JSON sonuçlar
├── requirements.txt
├── runtime.txt              # python-3.11
└── .streamlit/config.toml   # Tema
```

## 🖥️ Lokal Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Tarayıcı: http://localhost:8501

## ☁️ Streamlit Community Cloud Deploy

1. Bu repoyu GitHub'a push edin.
2. https://share.streamlit.io adresinde **New app** → repo + branch (`main`) seçin.
3. **Main file path:** `app/streamlit_app.py`
4. **Advanced settings → Python version:** 3.11
5. **Deploy** — ilk build ~3-5 dakika sürer, ardından `*.streamlit.app` URL'si üretilir.

## 🎛️ Uygulama Sekmeleri

- **🔎 Tek Model** — bir model seç, yorum gir, tahmin al.
- **⚖️ 3 Modeli Karşılaştır** — aynı yorumu 3 modele gönder, yan yana skorları gör.
- **📊 Model Analizi** — accuracy/precision/recall/F1 + training curves + confusion matrix.

## ⚠️ Notlar

- Modeller **yalnızca İngilizce** IMDB verisiyle eğitildi; doğruluk İngilizce yorumda en yüksektir.
- Transfer learning yok, tüm ağırlıklar sıfırdan öğrenildi.
- İlk tahmin model yüklemesi nedeniyle 5–10 sn sürebilir; sonrakiler <1 sn.
