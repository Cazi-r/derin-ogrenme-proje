# Model 3: Bidirectional LSTM

BIM 430 Derin Öğrenme Projesi. IMDB film yorumları üzerinde duygu analizi yapan projenin üçüncü modeli. BiLSTM metni hem ileri hem geri yönde işler, her kelime için geçmiş ve gelecek bağlamı dikkate alır.

## Veri Seti

Model 1 ve Model 2 ile aynı: Keras IMDB, 22.500 / 2.500 / 25.000 bölünme, VOCAB_SIZE=10.000, MAX_LEN=250, seed=42.

## Mimari

```
Input (250)
  → Embedding (10000 → 128)
  → Bidirectional LSTM (64 birim × 2 yön = 128)
  → Dropout(0.5) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

Referans: Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

## Eğitim

Model 1 ve Model 2 ile aynı hiperparametreler: Adam (lr=0.001), Binary Crossentropy, batch=64, max 15 epoch, early stopping patience=3, seed=42.

## Sonuçlar (Test Seti)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.8484 |
| Precision | 0.8434 |
| Recall | 0.8558 |
| F1-Score | 0.8495 |

## Notlar

- Transfer learning ve fine-tuning yoktur; embedding dahil tüm ağırlıklar sıfırdan öğrenildi.
- Model 1, 2 ve 3 aynı veri bölünmesi ve aynı hiperparametrelerle eğitildi.
