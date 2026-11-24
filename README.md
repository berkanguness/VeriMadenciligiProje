# Karşılaştırmalı Duygu Analizi (Sentiment Analysis)

BIM448 dersi proje ödevi. Kural tabanlı (VADER) ve derin öğrenme tabanlı (BERT/RoBERTa)
duygu analizi yöntemleri, İngilizce ve Türkçe metinler üzerinde Pozitif / Negatif / Nötr
sınıfları için karşılaştırılmıştır. LLM tabanlı yöntem ekip arkadaşı tarafından ayrıca uygulanmıştır.

**Ekip:** Aylin Naz AVCI, Berkan GÜNEŞ

## Veri Setleri

| Dil | Kaynak | Kullanılan örnek |
|---|---|---|
| İngilizce | [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (Kaggle, `Tweets.csv`) | Her sınıftan 1.000, toplam 3.000 |
| Türkçe | [winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) (Hugging Face, `test.csv`) | Her sınıftan 1.000, toplam 3.000 |

Örnekler `random_state=42` ile seçilmiştir. Tekrarlanan metinler önceden silinmiştir.
Tüm yöntemler aynı `data/en_dataset.csv` ve `data/tr_dataset.csv` dosyalarını kullanır.

## Yöntemler

**VADER (kural tabanlı).** `compound` skoru ≥ 0.05 ise Positive, ≤ -0.05 ise Negative,
arası Neutral. VADER yalnızca İngilizce çalıştığı için Türkçe metinler önce
`Helsinki-NLP/opus-mt-tr-en` modeliyle İngilizceye çevrilmiştir. Ek deney olarak Türkçe
metin çevrilmeden de VADER'a verilmiştir.

**BERT tabanlı.**
- İngilizce: `cardiffnlp/twitter-roberta-base-sentiment-latest` (Negative / Neutral / Positive)
- Türkçe: `kaixkhazaki/turkish-sentiment` (BERTurk tabanlı, Negative / Notr / Positive)

Modeller bu çalışmada yeniden eğitilmemiştir, hazır fine-tune edilmiş halleri kullanılmıştır.

## Proje Yapısı

```
├── data/
│   ├── raw/                     # indirilen ham veri (repoda yok)
│   ├── en_dataset.csv           # 3.000 İngilizce örnek
│   ├── tr_dataset.csv           # 3.000 Türkçe örnek
│   └── tr_dataset_translated.csv
├── results/                     # tahminler, summary.csv, confusion matrix resimleri
├── prepare_data.py              # ham veriden eşit örnekli CSV'ler üretir
├── 01_vader.py                  # VADER (EN + TR)
├── 02_bert.py                   # BERT (EN + TR)
└── 03_evaluate.py               # hata sayıları, accuracy, F1, confusion matrix
```

## Kurulum ve Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib vaderSentiment
pip install transformers torch sentencepiece sacremoses
```

Ham veriyi indirip `data/raw/` içine koyun: Kaggle'dan `Tweets.csv`, Hugging Face'ten
`test.csv`. Sonra sırayla:

```bash
python prepare_data.py
python 01_vader.py
python 02_bert.py
python 03_evaluate.py
```

İlk çalıştırmada modeller internetten indirilir. Kodlar Python 3.14 ve macOS (Apple Silicon)
üzerinde çalıştırılmıştır.

## Sonuçlar

Her yöntem için sınıf bazlı **yanlış tahmin sayıları** (her sınıfta 1.000 örnek içinden):

| Yöntem | Pozitif hata | Negatif hata | Nötr hata | Toplam hata | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| VADER (EN) | 90 | 567 | 690 | 1347 | 0.551 | 0.528 |
| VADER (TR, çeviri ile) | 165 | 483 | 395 | 1043 | 0.652 | 0.647 |
| VADER (TR, çevirisiz) | 909 | 973 | 7 | 1889 | 0.370 | 0.244 |
| BERT (EN) | 107 | 233 | 346 | 686 | 0.771 | 0.769 |
| BERT (TR) | 20 | 153 | 0 | 173 | 0.942 | 0.942 |
| LLM (EN) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| LLM (TR) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Confusion matrix görselleri `results/cm_*.png` dosyalarındadır. Tam tablo `results/summary.csv`.

## Notlar ve Sınırlılıklar

- Türkçe veri setindeki Nötr cümleler Wikipedia'dan alınmıştır. Bu cümleler duygu taşımadığı
  için Nötr sınıf yapay olarak kolay ayrılır (BERT TR'de Nötr hatası 0).
- İngilizce (havayolu tweetleri) ve Türkçe (ürün yorumları, tweetler vb. karışık kaynaklar)
  veri setleri farklı alanlardan geldiği için iki dil arasındaki fark, dil farkı ile alan
  farkını ayırt etmez.
- Türkçe BERT modeli aynı veri setiyle eğitilmiştir. Model kartındaki örnek sayılarından,
  veri setinin test bölümünün modelin doğrulama kümesi olarak kullanıldığı anlaşılmaktadır.
  Bu nedenle değerlendirme örnekleri tam bağımsız değildir ve BERT TR sonucu iyimser olabilir.
- VADER'ın Türkçe sonuçları çeviri kalitesinden de etkilenir. Çevirisiz VADER hemen her
  metne Nötr dediği için bu deney çevirinin gerekliliğini göstermektedir.
