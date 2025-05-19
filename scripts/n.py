import pandas as pd

df = pd.read_csv(
    "./merged_all.csv",
    encoding="utf-8",
)

categories = []
for index, row in df.iterrows():
    if row["category"] not in categories:
        categories.append(row["category"])

print(categories)