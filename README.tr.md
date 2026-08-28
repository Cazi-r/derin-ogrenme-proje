<div align="center">

# IMDB Duygu Analizi

**IMDB film yorumlarında duygu sınıflandırması için üç derin öğrenme mimarisinin karşılaştırması ve canlı Streamlit demosu.**

[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-uygulama-FF4B4B.svg)](https://streamlit.io/)

[English](README.md) · [Türkçe](README.tr.md)

</div>

---

## Genel Bakış

Özel olarak tasarlanmış bir **Multi-Kernel TextCNN** modeli, literatürden iki referans model —
**LSTM** ve **Çift Yönlü LSTM (BiLSTM)** — ile IMDB ikili duygu veri seti (25.000 test yorumu)
üzerinde karşılaştırılmıştır. Üç model de sıfırdan eğitilmiştir; transfer öğrenme veya
ön-eğitimli embedding kullanılmamıştır.

Streamlit uygulaması ile kendi metninizi herhangi bir modele verebilir, üç modeli yan yana
karşılaştırabilir, eğitim grafiklerini ve karmaşıklık matrislerini inceleyebilirsiniz.

## Sonuçlar

25.000 yorumluk IMDB test kümesinin tamamı üzerinde değerlendirilmiştir.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Multi-Kernel TextCNN** (özel) | **0,8710** | 0,8725 | 0,8690 | **0,8707** |
| Çift Yönlü LSTM | 0,8484 | 0,8434 | 0,8558 | 0,8495 |
| LSTM | 0,8387 | 0,8186 | 0,8703 | 0,8437 |

Özel TextCNN modeli her metrikte öndedir. Konvolüsyonel n-gram öznitelikleri bu görevde
tekrarlayan (recurrent) mimarilerden hem daha başarılı hem de eğitimi çok daha ucuz olmuştur.

> [!NOTE]
> Üç model de aşırı öğrenme (overfitting) göstermektedir: TextCNN eğitimde %97,7 doğruluğa
> ulaşırken doğrulamada %87,0'de kalmaktadır. Doğrulama kaybı 2. epoch civarında dibe vurup
> sonrasında yükselmektedir. İlk eklenecek şey erken durdurma (early stopping) veya daha güçlü
> düzenlileştirme olmalıdır.

## Mimariler

Üç modelde de ortak:

| Hiperparametre | Değer |
|---|---|
| Sözlük boyutu | 10.000 |
| Maksimum dizi uzunluğu | 250 |
| Embedding boyutu | 128 |
| Batch size | 64 |
| Optimizer | Adam (lr = 0,001) |
| Seed | 42 |

| Model | Ayırt edici yapı | Epoch |
|---|---|---:|
| Multi-Kernel TextCNN | Farklı kernel genişliklerinde paralel konvolüsyonlar | 5 |
| LSTM | 128 LSTM birimi | 10 |
| Çift Yönlü LSTM | Yön başına 64 birim (128 çıkış boyutu) | 5 |

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Uygulama `http://localhost:8501` adresinde açılır.

> [!NOTE]
> İlk tahmin, modeller yüklendiği için 5–10 saniye sürer; sonrakiler 1 saniyenin altındadır.

### Streamlit Community Cloud'a dağıtım

1. [share.streamlit.io](https://share.streamlit.io) → **New app**, bu repo ve `main` branch'i seçin.
2. **Main file path:** `app/streamlit_app.py`
3. **Advanced settings → Python version:** 3.11
4. **Deploy.** İlk derleme 3–5 dakika sürer.

## Uygulama Sekmeleri

| Sekme | İşlevi |
|---|---|
| **Tek Model** | Bir model seç, yorum gir, tahmin al. |
| **3 Modeli Karşılaştır** | Aynı yorumu üç modele gönder, skorları yan yana gör. |
| **Model Analizi** | Accuracy / precision / recall / F1, eğitim grafikleri, karmaşıklık matrisleri. |

## Proje Yapısı

```
.
├── app/
│   ├── streamlit_app.py     Ana arayüz (3 sekme)
│   ├── model_loader.py      Üç modelin cache'li yüklenmesi
│   └── preprocess.py        IMDB tokenizasyon + padding
├── model1/                  Multi-Kernel TextCNN — ağırlık, notebook, grafik, sonuç JSON
├── model2/                  LSTM — aynı düzen
├── model3/                  Çift Yönlü LSTM — aynı düzen
├── docs/
│   ├── project-brief.pdf
│   └── presentation-outline.md
├── requirements.txt
├── runtime.txt              python-3.11
└── .streamlit/config.toml   Tema
```

Her `modelN/` klasörü kendi eğitilmiş `.keras` ağırlıklarını, eğitim notebook'unu, eğitim
grafiklerini, karmaşıklık matrisini ve yukarıdaki metrikleri içeren `results_model_N.json`
dosyasını barındırır.

## Sınırlamalar

- Yalnızca **İngilizce** yorumlarla eğitilmiştir — başka dillerde doğruluk düşer.
- Transfer öğrenme yoktur; tüm ağırlıklar sıfırdan öğrenilmiştir.
- Sözlük en sık geçen 10.000 token ile sınırlıdır, nadir kelimeler `<UNK>` olarak işlenir.

## Ders

**BIM-430 Derin Öğrenme** dönem projesi.

## Lisans

[MIT Lisansı](LICENSE) ile yayımlanmıştır.
