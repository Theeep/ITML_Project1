import pandas as pd

df = pd.read_csv("crimes_processed.csv")
neighbourhood_one_hot = pd.get_dummies(df["NEIGHBOURHOOD"])
weekeday_one_hot = pd.get_dummies(df["WEEKDAY"])

df = df.drop(["HUNDRED", "BLOCK", "NEIGHBOURHOOD", "WEEKDAY"], axis=1)
df = df.join(neighbourhood_one_hot)
df = df.join(weekeday_one_hot)

print(len(df.columns))

df.to_csv("crimes_processed_kNN.csv", encoding='utf-8', index=False)