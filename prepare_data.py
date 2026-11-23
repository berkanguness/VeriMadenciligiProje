from pathlib import Path
import pandas as pd

SEED = 42
PER_CLASS = 1000
LABELS = ["Positive", "Negative", "Neutral"]
LABEL_MAP = {"positive": "Positive", "negative": "Negative",
             "neutral": "Neutral", "notr": "Neutral", "nötr": "Neutral"}

DATA = Path("data")


def sample_balanced(df, n):
    parts = []
    for lab in LABELS:
        sub = df[df["label"] == lab]
        if len(sub) < n:
            raise ValueError(f"{lab} sınıfında sadece {len(sub)} örnek var")
        parts.append(sub.sample(n=n, random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def normalize_labels(series):
    out = series.astype(str).str.strip().str.lower().map(LABEL_MAP)
    if out.isna().any():
        raise ValueError(f"Tanınmayan etiketler: {series[out.isna()].unique()[:10]}")
    return out


# ---- İngilizce ----
en = pd.read_csv(DATA / "raw" / "Tweets.csv")
print("EN sütunlar:", list(en.columns))
en = en.rename(columns={"airline_sentiment": "label"})[["text", "label"]]
en["label"] = normalize_labels(en["label"])
en = en.drop_duplicates(subset="text")
print(en["label"].value_counts())
sample_balanced(en, PER_CLASS).to_csv(DATA / "en_dataset.csv", index=False)

# ---- Türkçe (test.csv'den alıyoruz) ----
tr = pd.read_csv(DATA / "raw" / "test.csv")
print("TR sütunlar:", list(tr.columns))
tr["label"] = normalize_labels(tr["label"])
tr = tr.drop_duplicates(subset="text")
print(tr["label"].value_counts())
sample_balanced(tr, PER_CLASS)[["text", "label"]].to_csv(DATA / "tr_dataset.csv", index=False)

print("Bitti: data/en_dataset.csv ve data/tr_dataset.csv oluştu")