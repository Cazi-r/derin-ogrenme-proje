# Model 1: Custom Multi-Kernel TextCNN

**BIM 430 — Derin Öğrenme Projesi**

---

## 📌 Proje Özeti

Bu klasör, IMDB film yorumları üzerinde **duygu analizi (sentiment analysis)** yapan projenin birinci modelini içerir. Model tamamen sıfırdan tasarlanmış olup, herhangi bir ön-eğitilmiş (pretrained) ağırlık veya transfer learning kullanılmamıştır.

---

## 📊 Veri Seti

| Özellik | Değer |
|---------|-------|
| **Kaynak** | Keras IMDB Dataset (Maas et al., 2011) |
| **Görev** | İkili sınıflandırma (Pozitif / Negatif yorum) |
| **Toplam Örnek** | 50.000 film yorumu |
| **Eğitim** | 22.500 örnek |
| **Validation** | 2.500 örnek |
| **Test** | 25.000 örnek |
| **Sınıf Dengesi** | Dengeli (50/50) |
| **Dil** | İngilizce |

### Ön İşleme
- En sık kullanılan **10.000 kelime** ile kelime dağarcığı oluşturuldu.
- Tüm yorumlar **250 token** uzunluğuna pad/truncate edildi.
- Veriler Keras tarafından zaten sayısal indekslere dönüştürülmüş durumda.

---

## 🧠 Model Mimarisi: Multi-Kernel TextCNN

Input (250 token)
↓
Embedding (10000 → 128 boyut, sıfırdan eğitilir)
↓
┌──────────────┬──────────────┬──────────────┐
│ Conv1D k=3   │ Conv1D k=4   │ Conv1D k=5   │
│ 64 filtre    │ 64 filtre    │ 64 filtre    │
│ ReLU         │ ReLU         │ ReLU         │
└──────┬───────┴──────┬───────┴──────┬───────┘
↓              ↓              ↓
GlobalMaxPool  GlobalMaxPool  GlobalMaxPool
└──────────────┼──────────────┘
↓
Concatenate (192 dim)
↓
Dropout (0.5)
↓
Dense (64, ReLU)
↓
Dropout (0.3)
↓
Dense (1, Sigmoid)
↓
Output

### Tasarım Gerekçeleri

| Bileşen | Seçim | Gerekçe |
|---------|-------|---------|
| **Embedding** | 128 boyut, sıfırdan | Transfer learning yasağı gereği pretrained kullanılmadı |
| **Paralel Conv (k=3,4,5)** | 3 farklı kernel | Farklı uzunluktaki n-gram kalıplarını yakalar (kısa, orta, uzun ifadeler) |
| **Filtre sayısı** | 64 | T4 GPU'da verimli, yeterli kapasite |
| **Aktivasyon** | ReLU | Gradient vanishing'e dayanıklı, hızlı |
| **GlobalMaxPooling** | Her filtre için | Duygu belirten kelimenin konum bağımsız en güçlü sinyalini alır |
| **Dropout (0.5 + 0.3)** | Çift katmanlı | Overfitting'e karşı iki aşamalı savunma |
| **Çıkış aktivasyonu** | Sigmoid | İkili sınıflandırma için standart |

---

## ⚙️ Eğitim Yapılandırması

| Parametre | Değer |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Batch Size | 64 |
| Max Epochs | 15 |
| Early Stopping | patience=3, restore_best_weights=True |
| Random Seed | 42 |
| GPU | NVIDIA T4 (Google Colab) |

---

## 📈 Sonuçlar

### Test Seti Metrikleri

| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.8710 |
| **Precision** | 0.8725 |
| **Recall** | 0.8690 |
| **F1-Score** | 0.8707 |

### Eğitim Süreci
- Eğitim **5. epoch'ta early stopping** ile durduruldu.
- **En iyi model 2. epoch'ta** elde edildi (val_loss=0.2927, val_accuracy=0.8796).
- 2. epoch'tan sonra validation loss artmaya başladı (overfitting sinyali).
- Dropout katmanları overfitting'i geciktirmeyi başardı.

### Yorum
Sınıflar arası metriklerin birbirine çok yakın olması (~%87), modelin dengeli çalıştığını ve herhangi bir sınıfa meyilli olmadığını gösterir. Sıfırdan eğitilmiş bir TextCNN için bu sonuç literatürle uyumludur (%86-89 aralığı).

---

## 📁 Dosya Yapısı

model1/
├── README.md                          # Bu dosya
├── model1_custom_textcnn.ipynb        # Ana notebook
├── model_1_custom_textcnn.keras       # Eğitilmiş model
├── results_model_1.json               # Tüm metrikler ve history
├── model1_training_curves.png         # Loss/Accuracy grafikleri
├── model1_confusion_matrix.png        # Confusion matrix
└── dataset_overview.png               # Veri seti istatistikleri

---

## 🔒 Tekrarlanabilirlik (Reproducibility)

- `numpy`, `tensorflow` ve `random` için **seed = 42** sabitlenmiştir.
- Train/Validation bölünmesi deterministiktir.
- Aynı seed ile çalıştırıldığında tüm modellerde aynı veri bölünmesi kullanılır.
- Bu sayede Model 2 (LSTM) ve Model 3 (BiLSTM) ile adil karşılaştırma yapılabilir.

---

## ⚠️ Önemli Notlar

- **Transfer learning kullanılmamıştır.** Embedding katmanı dahil tüm ağırlıklar sıfırdan öğrenilmiştir.
- **Fine-tuning yapılmamıştır.** Hazır model ağırlıkları üzerine ince ayar yoktur.
- Model tamamen öğrenciler tarafından tasarlanmıştır; literatürdeki TextCNN yaklaşımından ilham alınmış ancak katman sayıları, filtre sayıları, dropout oranları ve genel yapı grup tarafından belirlenmiştir.

---

## 🔗 Sonraki Adımlar

Projenin tamamlanması için aynı veri seti üzerinde aşağıdaki modeller de eğitilecek ve karşılaştırılacaktır:

1. ✅ **Model 1:** Custom Multi-Kernel TextCNN (bu klasör)
2. ⏳ **Model 2:** LSTM (literatür mimarisi, sıfırdan eğitim)
3. ⏳ **Model 3:** Bidirectional LSTM (literatür mimarisi, sıfırdan eğitim)

Her üç model aynı train/val/test bölünmesi üzerinde eğitilecek ve Accuracy, Precision, Recall, F1-Score metrikleriyle karşılaştırılacaktır.
