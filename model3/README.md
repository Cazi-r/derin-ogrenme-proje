# Model 3: Bidirectional LSTM (Literatür Mimarisi)

**BIM 430 — Derin Öğrenme Projesi**

---

## 📌 Proje Özeti

Bu klasör, IMDB film yorumları üzerinde duygu analizi yapan projenin **üçüncü modelini** içerir. Model literatürde yaygın kullanılan **Bidirectional LSTM (BiLSTM)** mimarisidir ve sıfırdan eğitilmiştir.

BiLSTM, metni hem ileri (soldan sağa) hem de geri (sağdan sola) yönlerde işleyerek, her kelime için hem geçmiş hem de gelecek bağlamı dikkate alır.

---

## 📊 Veri Seti

Model 1 ve Model 2 ile **birebir aynı** veri seti ve ön işleme kullanılmıştır:
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
Bidirectional LSTM (64 birim × 2 yön = 128 boyut)
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
Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

---

## ⚙️ Eğitim Yapılandırması

Model 1 ve Model 2 ile birebir aynı hiperparametreler:
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
| **Accuracy** | 0.8484 |
| **Precision** | 0.8434 |
| **Recall** | 0.8558 |
| **F1-Score** | 0.8495 |

---

## 📁 Dosyalar

- `model3_bilstm.ipynb` — Ana notebook
- `model_3_bilstm.keras` — Eğitilmiş model
- `results_model_3.json` — Metrikler ve history
- `model3_training_curves.png` — Loss/Accuracy grafikleri
- `model3_confusion_matrix.png` — Confusion matrix
- `model3_architecture.png` — Mimari görseli

---

## ⚠️ Önemli

- Transfer learning KULLANILMAMIŞTIR.
- Fine-tuning YAPILMAMIŞTIR.
- Embedding dahil tüm ağırlıklar sıfırdan öğrenilmiştir.
- Model 1, Model 2 ve Model 3 aynı veri kümesi ve aynı hiperparametrelerle eğitilmiştir.
