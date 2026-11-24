import pandas as pd
for n in ["en", "tr"]:
    d = pd.read_csv(f"results/llm_{n}.csv"); o = pd.read_csv(f"data/{n}_dataset.csv")
    print(n, len(d), "bos pred:", d.pred.isna().sum(),
          "| metinler ayni:", d.text.tolist() == o.text.tolist(),
          "| etiketler ayni:", d.label.tolist() == o.label.tolist())