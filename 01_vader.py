from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def compound(text):
    return analyzer.polarity_scores(str(text))["compound"]


def to_label(c):
    if c >= 0.05:
        return "Positive"
    if c <= -0.05:
        return "Negative"
    return "Neutral"


def run_vader(df, text_col, out_path):
    out = df.copy()
    out["compound"] = out[text_col].apply(compound)
    out["pred"] = out["compound"].apply(to_label)
    out.to_csv(out_path, index=False)
    print(out_path, "accuracy:", round((out["label"] == out["pred"]).mean(), 3))



en = pd.read_csv(DATA / "en_dataset.csv")
run_vader(en, "text", RESULTS / "vader_en.csv")


tr = pd.read_csv(DATA / "tr_dataset.csv")
cache = DATA / "tr_dataset_translated.csv"

if cache.exists():
    tr_t = pd.read_csv(cache, dtype={"translated": str})
else:
    name = "Helsinki-NLP/opus-mt-tr-en"
    tok = AutoTokenizer.from_pretrained(name)
    mt = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mt = mt.to(device)
    texts = [" ".join(str(t).split()) for t in tr["text"]]
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = mt.generate(**enc, max_new_tokens=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(len(out), "/", len(texts))
    tr_t = tr.copy()
    tr_t["translated"] = out
    tr_t.to_csv(cache, index=False)

tr_t["translated_final"] = tr_t["translated"].fillna("")
run_vader(tr_t, "translated_final", RESULTS / "vader_tr.csv")

run_vader(tr, "text", RESULTS / "vader_tr_ceviri_yok.csv")