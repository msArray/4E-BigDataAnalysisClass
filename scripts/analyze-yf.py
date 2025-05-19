import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import linregress
import os

plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォント

# CSVファイルを読み込む
df = pd.read_csv('./CNYJPY=X.csv', encoding="utf-8", skiprows=3)

df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

# NaT と NaN を除外
df = df.dropna(subset=['Date', 'Close'])
df = df.dropna(subset=['Date', 'High'])
df = df.dropna(subset=['Date', 'Low'])
df = df.dropna(subset=['Date', 'Open'])
df = df.dropna(subset=['Date', 'Volume'])

# 日付でソート（任意）
df = df.sort_values('Date')
# df = df.reset_index(drop=True)
# df = df.drop(columns=['index'])
# df = df.drop(columns=['Date'])
# df = df.drop(columns=['Close'])
# df = df.drop(columns=['High'])
# df = df.drop(columns=['Low'])
# df = df.drop(columns=['Open'])
# df = df.drop(columns=['Volume'])


# 日付を整数に変換（x軸として使用する）
df['date_int'] = df['Date'].apply(lambda x: (x - datetime(1970, 1, 1)).days)
# df['date_int'] = df['Date'].apply(lambda x: (x - datetime(1970, 1, 1)).days)
base_date = df['date_int'].min()
print(f"基準日付: {base_date}")

# 基準日付からの経過日数を計算
df['days_from_base'] = (df['date_int'] - base_date)
# 回帰直線の計算
slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['Close'])
# slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['High'])
# slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['Low'])
# slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['Open'])
# slope, intercept, r_value, p_value, std_err = linregress(df['days_from_base'], df['Volume'])

# 回帰直線を計算
regression_line = slope * df['days_from_base'] + intercept
# regression_line = slope * df['days_from_base'] + intercept
# regression_line = slope * df['days_from_base'] + intercept
# regression_line = slope * df['days_from_base'] + intercept
# regression_line = slope * df['days_from_base'] + intercept

# プロット（点のみ）
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], 'o', label='終値', markersize=2)
# plt.plot(df['Date'], df['High'], 'o', label='High Price', markersize=2)
# plt.plot(df['Date'], df['Low'], 'o', label='Low Price', markersize=2)
# plt.plot(df['Date'], df['Open'], 'o', label='Open Price', markersize=2)
# plt.plot(df['Date'], df['Volume'], 'o', label='Volume', markersize=2)
# 回帰直線をプロット
plt.plot(df['Date'], regression_line, color='red', label='回帰直線', linestyle='-')
plt.title('中国・元/円 (CNYJPY=X) の為替変化(終値)')
plt.xlabel('日付')
plt.ylabel('円')
plt.xlim(df['Date'].min(), df['Date'].max())
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月%d日'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.gcf().autofmt_xdate()
plt.show()


# 回帰直線の傾き、切片、p値、決定係数を表示
print(f"回帰直線の傾き: {slope}, 切片: {intercept}, p値: {p_value}, 決定係数: {r_value**2}")

# 別のアプローチ：日付を表示するが、切片を強調表示する方法
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], 'o', label='Close Price', markersize=2)
# plt.plot(df['Date'], df['High'], 'o', label='High Price', markersize=2)
# plt.plot(df['Date'], df['Low'], 'o', label='Low Price', markersize=2)
# plt.plot(df['Date'], df['Open'], 'o', label='Open Price', markersize=2)
# plt.plot(df['Date'], df['Volume'], 'o', label='Volume', markersize=2)

# 回帰直線をプロット
# plt.plot(df['Date'], regression_line, color='red', label=f'回帰直線 (y = {slope:.6f}x + {intercept:.2f})', linestyle='-')

# 日付のフォーマットを設定
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月%d日'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 3ヶ月ごとに目盛り
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 1日ごとに目盛り
# plt.gca().xaxis.set_major_locator(mdates.YearLocator(interval=1))  # 1年ごとに目盛り

# 切片を強調表示
# plt.axhline(y=intercept, color='blue', linestyle='--', label=f'切片: {intercept:.2f}円')


# X軸のラベルの回転を調整
plt.xticks(rotation=45)
plt.xlim(df['Date'].min(), df['Date'].max())
plt.xlabel('日付')
plt.ylabel('価格（円）')
plt.title('フィリピン・ペソ/円 (PHPJPY=X) の為替変化')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

