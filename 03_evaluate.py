from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, f1_score)

LABELS = ["Positive", "Negative", "Neutral"]
rows = []

for f in sorted(Path("results").glob("*.csv")):
    if f.name == "summary.csv":
        continue
    df = pd.read_csv(f).dropna(subset=["pred"])
    cm = confusion_matrix(df["label"], df["pred"], labels=LABELS)
    errors = cm.sum(axis=1) - cm.diagonal()   # her gerçek sınıf için yanlış sayısı
    rows.append({
        "yontem": f.stem,
        "Pozitif_hata": errors[0],
        "Negatif_hata": errors[1],
        "Notr_hata": errors[2],
        "Toplam_hata": errors.sum(),
        "Accuracy": round(accuracy_score(df["label"], df["pred"]), 3),
        "Macro_F1": round(f1_score(df["label"], df["pred"], labels=LABELS, average="macro"), 3),
    })
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(f.stem)
    plt.savefig(f"results/cm_{f.stem}.png", dpi=150)
    plt.close()

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv("results/summary.csv", index=False)