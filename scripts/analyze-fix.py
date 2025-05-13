import pandas as pd
import matplotlib.pyplot as plt
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

# NaT と NaN を除外
df = df.dropna(subset=['date', 'price'])

# 日付でソート（任意）
df = df.sort_values('date')

# 基準日付を設定（データの最初の日付）
base_date = df['date'].min()
print(f"基準日付: {base_date}")

# 基準日付からの経過日数を計算
df['days_from_base'] = (df['date'] - base_date).dt.days

# 回帰直線の計算（基準日からの日数を使用）
slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['price'])

# 回帰直線を計算
regression_line = slope * df['days_from_base'] + intercept

# プロット（点のみ）
plt.figure(figsize=(12, 5))
plt.scatter(df['days_from_base'], df['price'], color='green', label='値段', alpha=0.1, s=10)

# 回帰直線をプロット
plt.plot(df['days_from_base'], regression_line, color='red', label='回帰直線', linestyle='-')
print(f"回帰直線の傾き: {slope}, 切片: {intercept}, p値: {p_value}, 決定係数: {r_value**2}")

# X軸のラベルをカスタマイズ（主要な日付を表示）
def days_to_date_label(days):
    # numpy.int64 を Python の int に変換
    days_int = int(days)
    return (base_date + timedelta(days=days_int)).strftime('%Y年%m月')

# 主要な目盛りの位置を設定（例：6ヶ月ごと）
major_ticks = np.arange(0, max(df['days_from_base']) + 180, 180)
plt.xticks(major_ticks, [days_to_date_label(days) for days in major_ticks], rotation=45)

plt.xlabel('日付（{0}からの経過）'.format(base_date.strftime('%Y年%m月%d日')))
plt.ylabel('価格（円）')
plt.title('ガシャポン価格の時系列推移と回帰直線')
plt.grid(True)
plt.tight_layout()
plt.legend()

# グラフの余白を調整して切片がはっきり見えるようにする
plt.xlim(-5, max(df['days_from_base']) + 5)

plt.show()

# 別のアプローチ：日付を表示するが、切片を強調表示する方法
plt.figure(figsize=(12, 5))
plt.scatter(df['date'], df['price'], color='green', label='値段', alpha=0.1, s=10)

# 回帰直線をプロット
plt.plot(df['date'], regression_line, color='red', label=f'回帰直線 (y = {slope:.6f}x + {intercept:.2f})', linestyle='-')

# 日付のフォーマットを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 6ヶ月ごとに目盛り

# 切片を強調表示
plt.axhline(y=intercept, color='blue', linestyle='--', label=f'切片: {intercept:.2f}円')

# X軸のラベルの回転を調整
plt.xticks(rotation=45)
plt.xlim(base_date - timedelta(days=30), df['date'].max() + timedelta(days=30))
plt.xlabel('日付')
plt.ylabel('価格（円）')
plt.title('ガシャポン価格の時系列推移と回帰直線')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# おまけ：年ごとの平均価格を棒グラフで表示
df['year'] = df['date'].dt.year
yearly_avg = df.groupby('year')['price'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(12, 5))
bars = plt.bar(yearly_avg['year'], yearly_avg['mean'], color='skyblue')

# 各バーの上に平均値と件数を表示
for bar, mean, count in zip(bars, yearly_avg['mean'], yearly_avg['count']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{mean:.1f}円\n({count}件)', 
             ha='center', va='bottom')

plt.ylim(0, yearly_avg['mean'].max() + 100)
plt.xlabel('年')
plt.ylabel('平均価格（円）')
plt.title('ガシャポン価格の年間平均値')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()