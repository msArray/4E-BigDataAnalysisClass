import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from scipy.stats import linregress

plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォント

df = pd.read_csv(
    "./merged_gasolin_merge.csv", encoding="utf-8", usecols=["year", "month", "price"]
)

# price列を数値に変換（無効なデータはNaNに変換）
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df["date"] = pd.to_datetime(
    df.apply(
        lambda row: f"{int(row['year']):04d}-{int(row['month']):02d}-01", axis=1
    )
)

# NaT と NaN を除外
df = df.dropna(subset=["date", "price"])
# 日付でソート（任意）
df = df.sort_values("date")
# 基準日付を設定（データの最初の日付）
base_date = df["date"].min()
print(f"基準日付: {base_date}")
# 基準日付からの経過日数を計算
df["days_from_base"] = (df["date"] - base_date).dt.days
# 回帰直線の計算（基準日からの日数を使用）
slope, intercept, r_value, p_value, std_err = linregress(
    df["days_from_base"], df["price"]
)
# 回帰直線の値を計算
df["trend"] = slope * df["days_from_base"] + intercept
# グラフの描画
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["price"], label="実際の価格", marker="o", markersize=3)
plt.plot(
    df["date"], df["trend"], label="回帰直線", color="red", linestyle="-", linewidth=2
)
plt.title("ガソリン価格の推移")
plt.xlabel("日付")
plt.ylabel("価格[円]")
plt.xlim(df["date"].min(), df["date"].max())
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
