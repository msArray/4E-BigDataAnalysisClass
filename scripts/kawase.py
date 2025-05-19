import pandas as pd

df = pd.DataFrame(columns=["Date", "CNY", "PHP", "VND"])

cny = pd.read_csv("./CNYJPY=X.csv", encoding="utf-8", skiprows=3, usecols=[0, 1])
cny.columns = ["Date", "CNY"]
cny["Date"] = pd.to_datetime(cny["Date"])
cny["CNY"] = pd.to_numeric(cny["CNY"], errors="coerce")

php = pd.read_csv("./PHPJPY=X.csv", encoding="utf-8", skiprows=3, usecols=[0, 1])
php.columns = ["Date", "PHP"]
php["Date"] = pd.to_datetime(php["Date"])
php["PHP"] = pd.to_numeric(php["PHP"], errors="coerce")

vnd = pd.read_csv("./VNDJPY=X.csv", encoding="utf-8", skiprows=3, usecols=[0, 1])
vnd.columns = ["Date", "VND"]
vnd["Date"] = pd.to_datetime(vnd["Date"])
vnd["VND"] = pd.to_numeric(vnd["VND"], errors="coerce")

# データのマージ
df = pd.merge(cny, php, on="Date", how="outer")
df = pd.merge(df, vnd, on="Date", how="outer")
# 日付でソート（任意）
df = df.sort_values("Date")
print(df)

df.to_csv(
    "./merged_kawase.csv",
    index=False,
    encoding="utf-8-sig",
)