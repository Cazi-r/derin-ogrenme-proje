# Model 2: LSTM

BIM 430 Derin Öğrenme Projesi. IMDB film yorumları üzerinde duygu analizi yapan projenin ikinci modeli. Literatürde yaygın kullanılan klasik LSTM mimarisi, sıfırdan eğitildi.

## Veri Seti

Model 1 ile aynı: Keras IMDB, 22.500 / 2.500 / 25.000 bölünme, VOCAB_SIZE=10.000, MAX_LEN=250, seed=42.

## Mimari

```
Input (250)
  → Embedding (10000 → 128)
  → LSTM (128 birim, dropout=0.2)
  → Dropout(0.5) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

Referans: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

## Eğitim

Model 1 ile aynı hiperparametreler: Adam (lr=0.001), Binary Crossentropy, batch=64, max 15 epoch, early stopping patience=3, seed=42.

## Sonuçlar (Test Seti)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.8387 |
| Precision | 0.8186 |
| Recall | 0.8703 |
| F1-Score | 0.8437 |

## Notlar

- Transfer learning ve fine-tuning yoktur; embedding dahil tüm ağırlıklar sıfırdan öğrenildi.
- Model 1, 2 ve 3 aynı veri bölünmesi ve aynı hiperparametrelerle eğitildi (adil karşılaştırma).
