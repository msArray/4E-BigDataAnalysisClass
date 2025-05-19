import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "MS Gothic"  # 日本語フォント

df = pd.read_csv(
    "./merged_all.csv",
    encoding="utf-8",
)

df.columns = ["Date", "Name", "Price", "Category", "CNY", "PHP", "VND", "Gasoline"]
print(df["Price"].max())

# 波線で省略された箱ひげ図を作成
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), 
                             gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})

# 上の部分（高い値の範囲）
sns.boxplot(x="Category", y="Price", data=df, ax=ax1)
ax1.set_ylim(6000, df["Price"].max() * 1.1)  # 上部のY軸範囲
ax1.set_xlabel('')  # X軸ラベルを非表示
ax1.set_ylabel('')  # Y軸ラベルを非表示

# 下の部分（低い値の範囲）
sns.boxplot(x="Category", y="Price", data=df, ax=ax2)
ax2.set_ylim(0, 2500)  # 下部のY軸範囲
ax2.set_xlabel('')  # X軸ラベルを非表示

# カテゴリラベルの回転
plt.xticks(rotation=90)

# 波線で省略を表現
d = .5  # 波線の間隔
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
# ax1.plot((-d, 1+d), (-d, -d), **kwargs)        # 下線

kwargs.update(transform=ax2.transAxes)
# ax2.plot((-d, 1+d), (1+d, 1+d), **kwargs)      # 上線

# タイトルとレイアウト調整
# plt.suptitle("カテゴリ別の価格分布", y=0.98, fontsize=16)
ax2.set_xlabel("Category")
f.text(0.04, 0.5, 'Price', va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()