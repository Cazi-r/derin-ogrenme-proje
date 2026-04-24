# Model 2: LSTM (Literatür Mimarisi)

**BIM 430 — Derin Öğrenme Projesi**

---

## 📌 Proje Özeti

Bu klasör, IMDB film yorumları üzerinde duygu analizi yapan projenin **ikinci modelini** içerir. Model literatürde yaygın kullanılan klasik **LSTM** (Long Short-Term Memory) mimarisidir ve sıfırdan eğitilmiştir.

---

## 📊 Veri Seti

Model 1 ile **birebir aynı** veri seti ve ön işleme kullanılmıştır:
- Keras IMDB Dataset
- 22.500 eğitim / 2.500 validation / 25.000 test
- VOCAB_SIZE = 10.000, MAX_LEN = 250
- SEED = 42 (deterministik bölünme)

---

## 🧠 Model Mimarisi

```
Input (250 token)
    ↓
Embedding (10000 → 128 boyut, sıfırdan)
    ↓
LSTM (128 birim, dropout=0.2)
    ↓
Dropout (0.5)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (1, Sigmoid)
```

### Literatür Kaynağı
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

---

## ⚙️ Eğitim Yapılandırması

Model 1 ile birebir aynı hiperparametreler:
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch Size: 64
- Max Epochs: 15
- Early Stopping: patience=3
- Seed: 42

---

## 📈 Sonuçlar

| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.8387 |
| **Precision** | 0.8186 |
| **Recall** | 0.8703 |
| **F1-Score** | 0.8437 |

---

## 📁 Dosyalar

- `model2_lstm.ipynb` — Ana notebook
- `model_2_lstm.keras` — Eğitilmiş model
- `results_model_2.json` — Metrikler ve history
- `model2_training_curves.png` — Loss/Accuracy grafikleri
- `model2_confusion_matrix.png` — Confusion matrix
- `model2_architecture.png` — Mimari görseli

---

## ⚠️ Önemli

- Transfer learning KULLANILMAMIŞTIR.
- Fine-tuning YAPILMAMIŞTIR.
- Embedding dahil tüm ağırlıklar sıfırdan öğrenilmiştir.
- Model 1, Model 2 ve Model 3 aynı veri kümesi ve aynı hiperparametrelerle eğitilmiştir (adil karşılaştırma).
