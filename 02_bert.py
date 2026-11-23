from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline

DATA = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TR_MODEL = "kaixkhazaki/turkish-sentiment"

LABEL_MAP = {"positive": "Positive", "negative": "Negative",
             "neutral": "Neutral", "notr": "Neutral", "nötr": "Neutral"}

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = -1
print("Cihaz:", DEVICE)


def preprocess_tweet(text):
    out = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


def run(model_name, csv_in, csv_out, preprocess=None):
    df = pd.read_csv(csv_in)
    texts = df["text"].astype(str).tolist()
    if preprocess:
        texts = [preprocess(t) for t in texts]
    clf = pipeline("text-classification", model=model_name, device=DEVICE)
    preds = clf(texts, batch_size=32, truncation=True, max_length=128)
    df["raw_label"] = [p["label"] for p in preds]
    df["score"] = [p["score"] for p in preds]
    df["pred"] = df["raw_label"].str.strip().str.lower().map(LABEL_MAP)
    if df["pred"].isna().any():
        print("UYARI tanınmayan etiket:", df.loc[df["pred"].isna(), "raw_label"].unique())
    df.to_csv(csv_out, index=False)
    print(csv_out, "accuracy:", round((df["label"] == df["pred"]).mean(), 3))


run(EN_MODEL, DATA / "en_dataset.csv", RESULTS / "bert_en.csv", preprocess_tweet)
run(TR_MODEL, DATA / "tr_dataset.csv", RESULTS / "bert_tr.csv")