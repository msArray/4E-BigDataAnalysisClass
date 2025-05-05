import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import linregress

plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォント

# CSVファイルを読み込む
df = pd.read_csv('./only_gacha.csv', encoding="utf-8")

# price列を数値に変換（無効なデータはNaNに変換）
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 日付変換関数
def get_week_start_date(row):
    try:
        year = int(row['year'])
        month = int(row['month'])
        week = int(row['week'])

        if year < 2018 and month < 7:
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

# NaT と NaN を除外
df = df.dropna(subset=['date', 'price'])

# 日付でソート（任意）
df = df.sort_values('date')

# 日付を整数に変換（x軸として使用する）
df['date_int'] = df['date'].apply(lambda x: (x - datetime(1970, 1, 1)).days)

# 回帰直線の計算
slope, intercept, r_value, p_value, std_err = linregress(df['date_int'], df['price'])

# 回帰直線を計算
regression_line = slope * df['date_int'] + intercept

# プロット（点のみ）
plt.figure(figsize=(12, 5))
plt.scatter(df['date'], df['price'], color='green', label='値段', alpha=0.3)

# 回帰直線をプロット
plt.plot(df['date'], regression_line, color='red', label='回帰直線', linestyle='-')

# 日付のフォーマットを設定（YYYY-MM-DD）
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月%d日'))

# X軸のラベルの回転を調整
plt.xticks(rotation=45)
plt.xlim(df['date'].min(), df['date'].max())
plt.xlabel('日付')
plt.ylabel('価格（円）')
plt.title('ガシャポン価格の時系列推移と回帰直線')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
