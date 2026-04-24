# 🎓 Sunum Şablonu — IMDB Sentiment Analysis

**BIM 430 Derin Öğrenme Projesi**
Toplam: 14 slayt, ~15-20 dakika

---

## Slayt 1 — Kapak
- Proje adı: *"IMDB Film Yorumları Üzerinde Duygu Analizi: Üç Derin Öğrenme Modelinin Karşılaştırılması"*
- Ders: BIM 430 Derin Öğrenme
- Grup üyeleri (soyadlarla)
- Tarih

---

## Slayt 2 — Problem Tanımı
- **Ne yapıyoruz?** Film yorumunu okuyup pozitif mi negatif mi olduğunu tahmin etmek (ikili sınıflandırma).
- **Neden önemli?** Müşteri geri bildirim analizi, içerik öneri sistemleri, sosyal medya madenciliği.
- **Girdi → Çıktı** örneği: bir cümle + "Pozitif / Negatif" etiketi.
- *Görsel:* örnek pozitif + örnek negatif yorum kutucuğu.

---

## Slayt 3 — Veri Seti
- Kaynak: Keras IMDB Dataset (Maas et al., 2011)
- 50.000 yorum: **22.500 train / 2.500 val / 25.000 test**
- Sınıf dengesi: 50/50 (dengeli)
- Dil: İngilizce
- *Görsel:* `model1/dataset_overview.png`

---

## Slayt 4 — Veri Ön İşleme
- Vocab size: en sık **10.000 kelime**
- Max length: **250 token** (pad/truncate)
- Keras zaten sayısal indekslere çevirmiş.
- Seed = 42 (tekrarlanabilirlik).

---

## Slayt 5 — Model 1: Custom Multi-Kernel TextCNN *(öğrenci tasarımı ✅)*
- Mimari şeması (README'deki ASCII veya yeniden çizin).
- Tasarım gerekçesi: 3 paralel conv (k=3,4,5) → farklı n-gram boylarındaki duygu ifadelerini yakalar.
- Katman sayıları, dropout (0.5 + 0.3), filtre sayıları **grup tarafından seçildi** (vurgulayın — hoca bunu arıyor).
- *Görsel:* mimari diyagramı.

---

## Slayt 6 — Model 2: LSTM (Literatür)
- Hochreiter & Schmidhuber 1997 referansı.
- Mimari: Embedding → LSTM(128) → Dropout → Dense.
- **"Neden literatürden?"** — ödev gereği, ama sıfırdan eğitildi.
- *Görsel:* `model2/model2_architecture.png`

---

## Slayt 7 — Model 3: Bidirectional LSTM (Literatür)
- Schuster & Paliwal 1997 referansı.
- Mimari: Embedding → BiLSTM(64×2) → Dropout → Dense.
- LSTM'den farkı: metni iki yönde birden işler (geçmiş + gelecek bağlam).
- *Görsel:* `model3/model3_architecture.png`

---

## Slayt 8 — Eğitim Yapılandırması (Adil Karşılaştırma)
Tablo halinde, tek sayfada:
- Optimizer: Adam (lr=0.001)
- Loss: Binary Crossentropy
- Batch Size: 64, Max Epochs: 15
- Early Stopping: patience=3
- Seed: 42
- GPU: Google Colab T4

**Vurgu:** Her üç model **tamamen aynı** hiperparametre ve veri bölünmesiyle eğitildi → adil karşılaştırma.
**Transfer learning YOK, fine-tuning YOK** (ödev kuralı).

---

## Slayt 9 — Eğitim Eğrileri (Loss & Accuracy)
Üçünü yan yana koyun:
- Sol: `model1/model1_training_curves.png`
- Orta: `model2/model2_training_curves.png`
- Sağ: `model3/model3_training_curves.png`

**Yorum:** TextCNN erken overfit etmiş (2. epoch'ta best), LSTM daha stabil ama doğruluğu daha düşük. Early stopping 3 modelde de çalışmış.

---

## Slayt 10 — Confusion Matrix Karşılaştırması
Yan yana üç confusion matrix:
- `model1/model1_confusion_matrix.png`
- `model2/model2_confusion_matrix.png`
- `model3/model3_confusion_matrix.png`

**Yorum:** Model 2 (LSTM) Recall yüksek ama Precision düşük → daha fazla "false positive" veriyor (negatif yorumları pozitif sanıyor). TextCNN dengeli.

---

## Slayt 11 — Sonuçlar: Sayısal Karşılaştırma ⭐ (en önemli slayt)

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|----:|
| **Custom TextCNN** | **0.8710** | **0.8725** | 0.8690 | **0.8707** |
| LSTM | 0.8387 | 0.8186 | **0.8703** | 0.8437 |
| BiLSTM | 0.8484 | 0.8434 | 0.8558 | 0.8495 |

Bunun **altına bar chart** koyun (4 metrik × 3 model). *Yoksa bu grafiği ürettirin — sunumun en vurucu slaytı.*

---

## Slayt 12 — Yorum ve Sonuç
- **TextCNN neden kazandı?** IMDB yorumlarında duygu genelde lokal n-gram'larda yoğun ("waste of time", "best movie ever"). Paralel kernel'ler bu kalıpları LSTM'in sıralı işleyişinden daha verimli yakalıyor.
- **BiLSTM > LSTM** — çift yönlü bağlam fayda sağlamış (~%1 artış).
- **Öğrenci tasarımı model, klasik literatür modellerini geçti** → sunumun güçlü vurgusu.
- *Gelecek çalışma:* GloVe embedding (pretrained'in yasak olduğunu unutmayın — "gelecek çalışma" olarak önerilebilir), attention mekanizması, daha büyük vocab.

---

## Slayt 13 — İş Bölümü + Kaynakça
**İş bölümü:**
- Grup üyesi 1: veri ön işleme + …
- Grup üyesi 2: Model 1 (TextCNN) tasarımı ve eğitimi
- Grup üyesi 3: Model 2 (LSTM) eğitimi
- Grup üyesi 4: Model 3 (BiLSTM) eğitimi + sunum/raporlama

**Kaynakça:**
- Maas et al., 2011 — IMDB veri seti
- Hochreiter & Schmidhuber, 1997 — LSTM
- Schuster & Paliwal, 1997 — Bidirectional RNN
- Kim, 2014 — TextCNN (ilham)

---

## Slayt 14 — Teşekkürler / Sorular

---

## 📌 Üretilmesi Tavsiye Edilen Ek Görseller

1. **Karşılaştırmalı bar chart** (Slayt 11 için) — 4 metrik × 3 model.
2. **Üst üste bindirilmiş val_loss eğrisi** (3 modelin val_loss'u aynı grafikte) — Slayt 9'a ek.

Her ikisi de kısa matplotlib kodu ile üretilebilir.

---

## ✅ Teslim Kontrol Listesi

- [ ] Sunum PPTX/PDF hazır
- [ ] Tüm notebook'lar ve `.keras` modelleri dahil
- [ ] README dosyaları güncel
- [ ] Sunum + kod zip'lendi (dosya adı: grup soyadları)
- [ ] Kampüs bilgi sistemine **01 Mayıs 2026, 23:59'a kadar** yüklendi
- [ ] Sunum günü grup üyeleri arasında iş bölümü netleştirildi
