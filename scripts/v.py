import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# 日本語フォントの設定
plt.rcParams["font.family"] = "MS Gothic"

# データの読み込み
df = pd.read_csv("./merged_all.csv", encoding="utf-8")
df.columns = ["Date", "Name", "Price", "Category", "CNY", "PHP", "VND", "Gasoline"]

# データの確認
print("データの概要:")
print(df.head())
print(f"\nデータの行数: {len(df)}")
print("\nデータの基本統計量:")
print(df.describe())

# Dateカラムを日付型に変換
df["Date"] = pd.to_datetime(df["Date"])

# 外れ値処理（IQR法）
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n外れ値の閾値 - 下限: {lower_bound}, 上限: {upper_bound}")
outliers = df[(df["Price"] < lower_bound) | (df["Price"] > upper_bound)]
print(f"外れ値の数: {len(outliers)}, 全体の {len(outliers)/len(df)*100:.2f}%")

# 外れ値をクリッピング
df_cleaned = df.copy()
df_cleaned["Price"] = df_cleaned["Price"].clip(lower=lower_bound, upper=upper_bound)

# 特徴量間の相関を確認
numeric_cols = ["Price", "CNY", "PHP", "VND", "Gasoline"]
correlation_matrix = df_cleaned[numeric_cols].corr()
print("\n相関行列:")
print(correlation_matrix)

# 相関行列のヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix, annot=True, cmap=sns.color_palette("Reds", 10), fmt=".2f"
)
plt.title("相関行列ヒートマップ")
plt.tight_layout()
plt.show()

# 数値型の特徴量を選択
X = df_cleaned[["CNY", "PHP", "VND", "Gasoline"]]
y = df_cleaned["Price"]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 重回帰モデルの作成と訓練
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 予測
y_pred = lr_model.predict(X_test)

# モデル評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n重回帰モデルの結果:")
print(f"平均二乗誤差 (MSE): {mse:.4f}")
print(f"決定係数 (R²): {r2:.4f}")

# 回帰係数
print("\n回帰係数:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"切片: {lr_model.intercept_:.4f}")

# 予測値と実測値のプロット
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.xlabel("実測値")
plt.ylabel("予測値")
plt.title("重回帰モデルによる実測値と予測値")
plt.tight_layout()
plt.show()

# 特徴量の重要度の可視化
plt.figure(figsize=(10, 6))
plt.bar(X.columns, lr_model.coef_)
plt.xlabel("特徴量")
plt.ylabel("係数")
plt.title("特徴量の重要度 (回帰係数)")
plt.tight_layout()
plt.show()

# 残差分析
residuals = y_test - y_pred

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="r", linestyles="--")
plt.xlabel("予測値")
plt.ylabel("残差")
plt.title("残差プロット")

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel("残差")
plt.title("残差の分布")
plt.tight_layout()
plt.show()

# 3D散布図で多変量関係を可視化 (上位2つの特徴量を選択)
# 相関が高い2つの特徴量を選択
top_features = correlation_matrix["Price"].abs().sort_values(ascending=False).index[1:3]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# データ点をプロット
ax.scatter(
    df_cleaned[top_features[0]],
    df_cleaned[top_features[1]],
    df_cleaned["Price"],
    c=df_cleaned["Price"],
    cmap="viridis",
    alpha=0.5,
)

# 回帰平面を作成
x_surf = np.linspace(df_cleaned[top_features[0]].min(), df_cleaned[top_features[0]].max(), 20)
y_surf = np.linspace(df_cleaned[top_features[1]].min(), df_cleaned[top_features[1]].max(), 20)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

# 回帰平面のZ値を計算
exog = pd.DataFrame({
    top_features[0]: x_surf.ravel(),
    top_features[1]: y_surf.ravel(),
    # 他の特徴量は平均値で固定
    **{col: np.mean(df_cleaned[col]) for col in X.columns if col not in top_features}
})[X.columns]

z_surf = lr_model.predict(exog).reshape(x_surf.shape)

# 回帰平面をプロット
ax.plot_surface(
    x_surf, y_surf, z_surf,
    cmap="viridis",
    alpha=0.3,
    edgecolor="none"
)

ax.set_xlabel(top_features[0])
ax.set_ylabel(top_features[1])
ax.set_zlabel("Price")
plt.title(f"重回帰平面: {top_features[0]} と {top_features[1]} の関係")
plt.tight_layout()
plt.show()

# 各特徴量と価格のペアプロット
sns.pairplot(
    df_cleaned,
    x_vars=["CNY", "PHP", "VND", "Gasoline"],
    y_vars=["Price"],
    kind="reg",
    height=4,
    aspect=1,
    plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.3}}
)
plt.suptitle("特徴量と価格の関係", y=1.02)
plt.tight_layout()
plt.show()

# 時系列での価格と特徴量の関係
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for i, feature in enumerate(["CNY", "PHP", "VND", "Gasoline"]):
    axes[i].plot(df_cleaned["Date"], df_cleaned[feature], label=feature, color="blue")
    ax2 = axes[i].twinx()
    ax2.plot(df_cleaned["Date"], df_cleaned["Price"], label="Price", color="red", alpha=0.5)
    
    axes[i].set_ylabel(feature, color="blue")
    ax2.set_ylabel("Price", color="red")
    axes[i].legend(loc="upper left")
    ax2.legend(loc="upper right")

plt.xlabel("Date")
plt.suptitle("時系列での特徴量と価格の関係")
plt.tight_layout()
plt.show()