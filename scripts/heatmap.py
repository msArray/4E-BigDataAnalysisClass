import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm

df = pd.read_csv(
    "./merged_all.csv",
    encoding="utf-8",
)

df.columns = ["Date", "Name", "Price", "Category", "CNY", "PHP", "VND", "Gasoline"]
print(df.head())

df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)  # 0,1列目の削除
df.dropna(how="any", inplace=True)  # Nanが含まれる行の削除


scaler = StandardScaler()  # 標準化を行う関数StandardScalerをscalerという名前に定義
dfs = scaler.fit_transform(df)  # データであるdfを標準化

corrs = np.corrcoef(dfs, rowvar=False)  # 相関係数の計算
sns.heatmap(
    corrs,
    cmap=sns.color_palette("Reds", 10),
    annot=True,
    xticklabels=df.columns,
    yticklabels=df.columns,
)  # ヒートマップで可視化
plt.show()  # ヒートマップの表示
