import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from scipy.stats import linregress
from datetime import datetime, timedelta
import numpy as np

plt.rcParams["font.family"] = "MS Gothic"  # 日本語フォント

# CSVファイルを読み込む
df = pd.read_csv("./merged.csv", encoding="utf-8")


# 日付変換関数
def get_week_start_date(row):
    try:
        year = int(row["year"])
        month = int(row["month"])
        week = int(row["week"])

        if year < 18 and month < 7:
            # 古いデータ：上旬中旬下旬の週番号に対応（仮に1:上旬, 2:中旬, 3:下旬として扱う）
            if week == 1:
                day = 5
            elif week == 2:
                day = 15
            else:
                day = 25
            return datetime(year + 2000, month, day)
        else:
            # 新しいデータ：週番号（1週目の月曜日 + 7日 * (week - 1)）
            first_day = datetime(year + 2000, month, 1)
            weekday = first_day.weekday()
            days_until_monday = (7 - weekday) % 7
            first_monday = first_day + timedelta(days=days_until_monday)
            return first_monday + timedelta(weeks=week - 1)
    except:
        return pd.NaT


# 日付列の作成
df["Date"] = df.apply(get_week_start_date, axis=1)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

# 為替レートのデータを読み込む
df_kawase = pd.read_csv(
    "./merged_kawase.csv", encoding="utf-8", usecols=["Date", "CNY", "PHP", "VND"]
)
df_kawase["Date"] = pd.to_datetime(
    df_kawase["Date"], format="%Y-%m-%d", errors="coerce"
)

# ガソリン価格のデータを読み込む
df_gasoline = pd.read_csv(
    "./merged_gasolin_merge.csv",
    encoding="utf-8",
    skiprows=1,
)
df_gasoline.columns = ["Year", "Month", "Gasoline"]

merged_df = pd.DataFrame(
    columns=["Date", "name", "price", "category", "CNY", "PHP", "VND", "Gasoline"]
)

for index, row in df.iterrows():
    # 日付がNaTの場合はスキップ
    if pd.isna(row["Date"]):
        continue

    # 日付に対応する為替レートを取得
    date = row["Date"]

    # まず完全一致を試みる
    kawase_row = df_kawase[df_kawase["Date"] == date]

    # 完全一致するデータがない場合、最も近い日付のデータを探す
    if kawase_row.empty:
        # 各日付との差分を計算
        df_kawase["date_diff"] = abs((df_kawase["Date"] - date).dt.days)

        # 日付の差が最小のレコードを取得
        closest_idx = df_kawase["date_diff"].idxmin()
        kawase_row = df_kawase.loc[[closest_idx]]

        # 後で削除するために一時カラムを削除
        df_kawase = df_kawase.drop(columns=["date_diff"])
    
    gasoline_row = df_gasoline[
        (df_gasoline["Year"] == date.year) & (df_gasoline["Month"] == date.month)
    ]

    # 利用するデータを作成
    merged_row = {
        "Date": date,
        "price": int(row["price"].replace(",", "")),
        "name": row["name"],
        "category": row["category"],
        "CNY": kawase_row.iloc[0]["CNY"],
        "PHP": kawase_row.iloc[0]["PHP"],
        "VND": kawase_row.iloc[0]["VND"],
        "Gasoline": gasoline_row.iloc[0]["Gasoline"] if not gasoline_row.empty else None,
    }
    merged_df = pd.concat([merged_df, pd.DataFrame([merged_row])], ignore_index=True)

merged_df["price"].rename("Price", inplace=True)
merged_df["name"].rename("Name", inplace=True)
merged_df["category"].rename("Category", inplace=True)

# 日付でソート
merged_df = merged_df.sort_values(by="Date")

merged_df.to_csv("./merged_all.csv", index=False, encoding="utf-8-sig")
# print(merged_df)
