import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import linregress


plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォント

# CSVファイルを読み込む
df = pd.read_csv('./without_other.csv', encoding="utf-8")

# price列を数値に変換（無効なデータはNaNに変換）
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 日付変換関数
def get_week_start_date(row):
    try:
        year = int(row['year'])
        month = int(row['month'])
        week = int(row['week'])

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
df['date'] = df.apply(get_week_start_date, axis=1)
df['date_int'] = df['date'].apply(lambda x: (x - datetime(1970, 1, 1)).days)

df_fi = pd.read_csv('./CNYJPY=X.csv', encoding="utf-8", skiprows=3)

df_fi.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df_fi['Close'] = pd.to_numeric(df_fi['Close'], errors='coerce')
df_fi['Date'] = pd.to_datetime(df_fi['Date'], format='%Y-%m-%d', errors='coerce')
df_fi = df_fi.dropna(subset=['Date', 'Close'])
df_fi['date_int'] = df_fi['Date'].apply(lambda x: (x - datetime(1970, 1, 1)).days)
base_date = min(df['date_int'].min(), df_fi['date_int'].min())
print(f"基準日付: {base_date}")

df['days_from_base'] = (df['date_int'] - base_date)
df_fi['days_from_base'] = (df_fi['date_int'] - base_date)

print(df['days_from_base'])

slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['price'])
slope_fi, intercept_fi, r_value_fi, p_value_fi, std_err_fi = linregress(df_fi['days_from_base'], df_fi['Close'])

# 回帰直線を計算
regression_line = slope * df['days_from_base'] + intercept
regression_line_fi = slope_fi * df_fi['days_from_base'] + intercept_fi

print(f"ガシャポン価格の回帰直線の傾き: {slope}, 切片: {intercept}, p値: {p_value}, 決定係数: {r_value**2}")
print(f"人民元/円レートの回帰直線の傾き: {slope_fi}, 切片: {intercept_fi}, p値: {p_value_fi}, 決定係数: {r_value_fi**2}")

# 日付ごとの平均価格を計算
df_avg = df.groupby('date')['price'].mean().reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))

# 左側のy軸 (ガシャポン価格用)
color1 = 'tab:blue'
ax1.set_xlabel('日付')
ax1.set_ylabel('ガシャポン価格（円）', color=color1)
# 個別データをプロット (小さい点で薄く表示)
ax1.plot(df['date'], df['price'], 'o', color=color1, label='個別価格', markersize=2, alpha=0.25)
# 日付ごとの平均価格を追加 (線と大きめの点でハイライト)
# ax1.plot(df_avg['date'], df_avg['price'], '-', color='green', label='日別平均価格', linewidth=1)
# ax1.plot(df_avg['date'], df_avg['price'], 'o', color='darkblue', markersize=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 900)
ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
ax1.plot(df['date_int'], regression_line, color='blue', label='ガシャポン価格 回帰直線', linestyle='-')
plt.xticks(rotation=45)

# 右側のy軸 (為替レート用)
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('人民元/円 為替レート', color=color2)
ax2.plot(df_fi['Date'], df_fi['Close'], linestyle='-', color=color2, label='人民元/円 為替レート', markersize=2)
ax2.plot(df_fi['Date'], regression_line_fi, color='orange', label='人民元/円 為替レート 回帰直線', linestyle='-')
ax2.tick_params(axis='y', labelcolor=color2)

# 人民元/円の値範囲を0から20に制限
ax2.set_ylim(0, 30)

# x軸の日付フォーマット設定
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月%d日'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# 凡例の設置
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('ガシャポン価格と人民元/円レートの推移')
plt.xlim(df['date'].min(), df['date'].max())
fig.tight_layout()  # レイアウトの自動調整
plt.show()