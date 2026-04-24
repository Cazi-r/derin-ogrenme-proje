# Model 1: Multi-Kernel TextCNN

BIM 430 Derin Öğrenme Projesi. IMDB film yorumları üzerinde duygu analizi yapan projenin ilk modeli. Sıfırdan tasarlanmış, pretrained ağırlık veya transfer learning kullanılmamıştır.

## Veri Seti

- Kaynak: Keras IMDB Dataset (Maas et al., 2011) — İngilizce, ikili sınıflandırma (pozitif/negatif)
- 22.500 eğitim / 2.500 validation / 25.000 test, sınıf dengesi 50/50
- En sık 10.000 kelime ile kelime dağarcığı, tüm yorumlar 250 token'a pad/truncate

## Mimari

```
Input (250)
  → Embedding (10000 → 128)
  → Conv1D k=3 (64) ┐
  → Conv1D k=4 (64) ├→ GlobalMaxPool ×3 → Concat (192)
  → Conv1D k=5 (64) ┘
  → Dropout(0.5) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```

Paralel Conv1D katmanlarıyla farklı uzunluktaki n-gram kalıpları yakalanır (k=3,4,5). GlobalMaxPooling duygu belirten kelimenin konum-bağımsız en güçlü sinyalini seçer.

## Eğitim

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Batch Size | 64 |
| Max Epochs | 15 |
| Early Stopping | patience=3, restore_best_weights=True |
| Seed | 42 |
| Ortam | NVIDIA T4 (Google Colab) |

Eğitim 5. epoch'ta early stopping ile durdu; en iyi model 2. epoch'ta elde edildi (val_loss=0.2927, val_accuracy=0.8796).

## Sonuçlar (Test Seti)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.8710 |
| Precision | 0.8725 |
| Recall | 0.8690 |
| F1-Score | 0.8707 |

## Notlar

- Transfer learning ve fine-tuning yoktur; embedding dahil tüm ağırlıklar sıfırdan öğrenildi.
- `numpy`, `tensorflow` ve `random` için seed=42 sabitlendi. Model 1, 2 ve 3 aynı train/val/test bölünmesi üzerinde eğitildi.
